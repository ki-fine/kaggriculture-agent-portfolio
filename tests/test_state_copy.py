"""Semantic tests for the extracted clone; no SDK import or game execution.

The small Struct fixture below is derived from kaggle_environments/utils.py.
Copyright 2020 Kaggle Inc. Licensed under Apache License, Version 2.0.
See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt.
"""
import copy
import copyreg
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import state_copy

# SDK behavior copied for isolated semantic tests; this is not an imported SDK.
STRUCT_FIXTURE = '''class Struct(dict):
    def __init__(self, **entries):
        entries = {k: v for k, v in entries.items() if k != "items"}
        dict.__init__(self, entries)
        self.__dict__.update(entries)
    def __setattr__(self, attr, value):
        self.__dict__[attr] = value
        self[attr] = value
'''


def graph_fingerprint(value):
    """Compare both views and graph topology, rather than mapping equality alone."""
    seen = {}

    def visit(obj):
        kind = type(obj)
        if kind in (type(None), bool, int, float, str, bytes):
            return (kind.__name__, obj)
        if id(obj) in seen:
            return ("alias", seen[id(obj)])
        tag = len(seen)
        seen[id(obj)] = tag
        if isinstance(obj, dict):
            attrs = visit(vars(obj)) if hasattr(obj, "__dict__") else None
            return (kind.__name__, tag, attrs,
                    tuple((visit(k), visit(v)) for k, v in obj.items()))
        if isinstance(obj, (list, tuple)):
            return (kind.__name__, tag, tuple(visit(v) for v in obj))
        if hasattr(obj, "__dict__"):
            return (kind.__name__, tag, visit(vars(obj)))
        return (kind.__name__, repr(obj))

    return visit(value)


class StateCopyTests(unittest.TestCase):
    def setUp(self):
        namespace = {"__name__": "kaggle_environments.utils"}
        exec(compile(STRUCT_FIXTURE, "<isolated-sdk-struct-fixture>", "exec",
                     dont_inherit=True), namespace)
        self.Struct = namespace["Struct"]
        self.previous_module = sys.modules.get("kaggle_environments.utils")
        sys.modules["kaggle_environments.utils"] = SimpleNamespace(Struct=self.Struct)
        state_copy._BUDGET_STRUCT_KINDS.clear()
        state_copy.COPY_STATS.update(fast=0, fallback=0, struct_nodes_attempted=0)

    def tearDown(self):
        copyreg.dispatch_table.pop(self.Struct, None)
        if self.previous_module is None:
            sys.modules.pop("kaggle_environments.utils", None)
        else:
            sys.modules["kaggle_environments.utils"] = self.previous_module

    def assert_same_as_deepcopy(self, value):
        before = graph_fingerprint(value)
        expected = copy.deepcopy(value)
        actual = state_copy.clone_state(value)
        self.assertEqual(graph_fingerprint(actual), graph_fingerprint(expected))
        self.assertEqual(graph_fingerprint(value), before, "Source object was mutated")
        return actual

    def test_dict_and_attribute_views_can_disagree(self):
        value = self.Struct(x=[1], gone=[2])
        value["x"] = [3]
        value["mapping_only"] = {"a": [4]}
        value.__dict__["attribute_only"] = {"b": [5]}
        del value["gone"]
        cloned = self.assert_same_as_deepcopy(value)
        self.assertEqual(cloned["x"], [3])
        self.assertEqual(cloned.x, [1])
        self.assertNotIn("gone", cloned)
        self.assertEqual(cloned.gone, [2])
        self.assertNotIn("attribute_only", cloned)
        self.assertNotIn("mapping_only", vars(cloned))
        cloned.x.append(9)
        self.assertEqual(value.x, [1])
        self.assertEqual(state_copy.COPY_STATS["fallback"], 0)

    def test_deleted_attribute_does_not_delete_mapping(self):
        value = self.Struct(x=[1])
        del value.__dict__["x"]
        cloned = self.assert_same_as_deepcopy(value)
        self.assertEqual(cloned["x"], [1])
        self.assertNotIn("x", vars(cloned))

    def test_shared_graph_across_mapping_attributes_and_outer_objects(self):
        shared = [{"count": 7}]
        value = self.Struct(left=shared)
        value.right = shared
        outer = {"struct": value, "attrs": vars(value), "shared": shared}
        cloned = self.assert_same_as_deepcopy(outer)
        node = cloned["struct"]
        self.assertIs(node.left, node["left"])
        self.assertIs(node.left, node.right)
        self.assertIs(node.left, cloned["shared"])
        # deepcopy reconstructs attrs with __dict__.update(state), so the outer
        # state-dict reference is distinct while nested object aliases remain.
        self.assertIsNot(vars(node), cloned["attrs"])
        self.assertIs(node.left, cloned["attrs"]["left"])
        self.assertIsNot(node.left, shared)

    def test_cycles_and_self_referential_attributes(self):
        value = self.Struct(x=1)
        value.self = value
        value.__dict__["attrs_self"] = value.__dict__
        cloned = self.assert_same_as_deepcopy(value)
        self.assertIs(cloned.self, cloned)
        self.assertIs(cloned["self"], cloned)
        # Match deepcopy's reconstruction exactly; its copied state dictionary
        # cycles to itself rather than becoming the new object's __dict__.
        self.assertIsNot(cloned.attrs_self, vars(cloned))
        self.assertIs(cloned.attrs_self["attrs_self"], cloned.attrs_self)

    def test_plain_json_graph_copies_mutable_nodes(self):
        shared = [1, {"nested": [2]}]
        value = {"left": shared, "right": shared}
        cloned = self.assert_same_as_deepcopy(value)
        self.assertIs(cloned["left"], cloned["right"])
        self.assertIsNot(cloned["left"], shared)

    def test_deepcopy_before_first_guard_empty_slot_cache(self):
        value = self.Struct(x=[1])
        self.assertNotIn("__slotnames__", self.Struct.__dict__)
        expected = copy.deepcopy(value)
        self.assertEqual(self.Struct.__dict__["__slotnames__"], [])
        self.assertEqual(state_copy._BUDGET_STRUCT_KINDS, {})
        cloned = state_copy.clone_state(value)
        self.assertEqual(graph_fingerprint(cloned), graph_fingerprint(expected))
        self.assertEqual(state_copy.COPY_STATS["fast"], 1)
        self.assertEqual(state_copy.COPY_STATS["fallback"], 0)
        self.assertGreater(state_copy.COPY_STATS["struct_nodes_attempted"], 0)

    def test_nonempty_slot_cache_rejected_even_after_positive_cache(self):
        self.assertTrue(state_copy._budget_is_struct(self.Struct))
        self.Struct.__slotnames__ = ["unexpected"]
        self.assertFalse(state_copy._budget_is_struct(self.Struct))
        self.Struct.__slotnames__ = ()
        self.assertFalse(state_copy._budget_is_struct(self.Struct))
        self.Struct.__slotnames__ = []
        self.assertTrue(state_copy._budget_is_struct(self.Struct))

    def test_custom_protocol_hook_uses_whole_object_fallback(self):
        value = self.Struct(x=1)
        value.__dict__["__deepcopy__"] = lambda memo: {"hook_result": 7}
        self.assertEqual(state_copy.clone_state(value), copy.deepcopy(value))
        self.assertEqual(state_copy.clone_state(value), {"hook_result": 7})
        self.assertEqual(state_copy.COPY_STATS["fast"], 0)
        self.assertEqual(state_copy.COPY_STATS["fallback"], 2)

    def test_registered_reducer_uses_whole_object_fallback(self):
        self.assertTrue(state_copy._budget_is_struct(self.Struct))
        copyreg.pickle(self.Struct, lambda value: (dict, ({"registered_reducer": 9},)))
        value = self.Struct(x=[1])
        self.assertEqual(state_copy.clone_state(value), copy.deepcopy(value))
        self.assertEqual(state_copy.COPY_STATS["fallback"], 1)

    def test_unsupported_nested_object_preserves_custom_hook_once(self):
        class Hook:
            calls = 0

            def __deepcopy__(self, memo):
                type(self).calls += 1
                result = type(self)()
                memo[id(self)] = result
                return result

        hook = Hook()
        value = self.Struct(left=hook, right=hook)
        cloned = state_copy.clone_state(value)
        self.assertIs(cloned.left, cloned.right)
        self.assertEqual(Hook.calls, 1)
        self.assertEqual(state_copy.COPY_STATS["fallback"], 1)

    def test_dict_subclass_is_not_silently_treated_as_sdk_struct(self):
        class OtherDict(dict):
            pass

        value = OtherDict(x=[1])
        self.assertIs(type(state_copy.clone_state(value)), OtherDict)
        self.assertEqual(state_copy.COPY_STATS["fallback"], 1)

    def test_extracted_helper_matches_pinned_bytes(self):
        source = Path(state_copy.__file__).read_text(encoding="utf-8")
        helper = source[source.index("import sys as _budget_sys"):
                        source.index("\n# Public aliases;")]
        self.assertEqual(hashlib.sha256(helper.encode()).hexdigest(),
                         state_copy.SOURCE_HELPER_SHA256)


if __name__ == "__main__":
    unittest.main()
