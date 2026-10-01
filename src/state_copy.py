"""Exact R7 clone helper extracted from the released observer module.

Keeps both mapping and attribute views, aliases and cycles. Unsupported objects
use whole-object copy.deepcopy; this module is not a complete Kaggle agent.
The embedded Struct reference below derives from kaggle_environments/utils.py:
Copyright 2020 Kaggle Inc, licensed under Apache License, Version 2.0.
See THIRD_PARTY_NOTICES.md and licenses/Apache-2.0.txt in this repository.
"""
import copy

SOURCE_RELEASE_SHA256 = '087a1ec47310d86da8f3f29a6ad394792558595b48053e4d346dfebc80dfa43b'
SOURCE_OBSERVER_SHA256 = '4672b7dcfacd517805e310a4a175b80deba59a6e1feebd652fbbaf56127d7d98'
SOURCE_HELPER_SHA256 = 'c6f1028c6fc81228d2b028d08915b7f0058264ce0a988150d87af83477c07150'

import sys as _budget_sys
_BUDGET_COPY_SCALARS=frozenset((type(None),bool,int,float,str,bytes))
_BUDGET_COPY_STATS={'fast':0,'fallback':0,'struct_nodes_attempted':0}
_BUDGET_STRUCT_KINDS={}
_BUDGET_STRUCT_REFERENCE_NAMESPACE={}
exec(compile('class Struct(dict):\n    def __init__(self, **entries):\n        entries = {k: v for k, v in entries.items() if k != "items"}\n        dict.__init__(self, entries)\n        self.__dict__.update(entries)\n    def __setattr__(self, attr, value):\n        self.__dict__[attr] = value\n        self[attr] = value\n','<verified-sdk-struct-reference>','exec',dont_inherit=True),_BUDGET_STRUCT_REFERENCE_NAMESPACE)
_BUDGET_STRUCT_REFERENCE=_BUDGET_STRUCT_REFERENCE_NAMESPACE['Struct']

class _BudgetCopyFallback(Exception):pass

def _budget_same_code(left,right):
    fields=('co_code','co_names','co_varnames','co_argcount','co_posonlyargcount',
            'co_kwonlyargcount','co_flags','co_freevars','co_cellvars')
    if any(getattr(left,k)!=getattr(right,k) for k in fields):return False
    if len(left.co_consts)!=len(right.co_consts):return False
    for a,b in zip(left.co_consts,right.co_consts):
        if type(a) is type(left):
            if type(b) is not type(right) or not _budget_same_code(a,b):return False
        elif type(a) is not type(b) or a!=b:return False
    return True

def _budget_is_struct(kind):
    slot_cache_ok=('__slotnames__' not in kind.__dict__
                   or (type(kind.__dict__['__slotnames__']) is list and not kind.__dict__['__slotnames__']))
    if kind in _BUDGET_STRUCT_KINDS:return slot_cache_ok and _BUDGET_STRUCT_KINDS[kind]
    module=_budget_sys.modules.get('kaggle_environments.utils')
    good=(module is not None and kind is getattr(module,'Struct',None)
          and kind.__module__=='kaggle_environments.utils' and kind.__name__=='Struct'
          and kind.__bases__==(dict,) and slot_cache_ok
          and set(_BUDGET_STRUCT_REFERENCE.__dict__)<=set(kind.__dict__)
          and set(kind.__dict__)-set(_BUDGET_STRUCT_REFERENCE.__dict__)<= {'__slotnames__'})
    if good:
        for name in ('__init__','__setattr__'):
            actual=kind.__dict__[name];expected=_BUDGET_STRUCT_REFERENCE.__dict__[name]
            if (type(actual) is not type(expected) or actual.__defaults__!=expected.__defaults__
                or actual.__kwdefaults__!=expected.__kwdefaults__
                or not _budget_same_code(actual.__code__,expected.__code__)):
                good=False;break
    _BUDGET_STRUCT_KINDS[kind]=good
    return good

def _budget_json_clone(value,memo):
    kind=type(value)
    if kind in _BUDGET_COPY_SCALARS:return value
    struct=(kind is not dict and kind is not list and _budget_is_struct(kind))
    if kind is not dict and kind is not list and not struct:raise _BudgetCopyFallback
    if struct and ('__deepcopy__' in value.__dict__ or '__reduce_ex__' in value.__dict__
                   or '__reduce__' in value.__dict__ or '__getstate__' in value.__dict__
                   or kind in copy.dispatch_table):raise _BudgetCopyFallback
    identity=id(value)
    if identity in memo:return memo[identity]
    if kind is list:
        result=[];memo[identity]=result
        for child in value:
            result.append(child if type(child) in _BUDGET_COPY_SCALARS else _budget_json_clone(child,memo))
    else:
        result=dict.__new__(kind) if struct else {}
        memo[identity]=result
        if struct:
            _BUDGET_COPY_STATS['struct_nodes_attempted']+=1
            # deepcopy reconstructs object state before copying dict items.
            result.__dict__.update(_budget_json_clone(value.__dict__,memo))
        for key,child in value.items():
            if type(key) not in _BUDGET_COPY_SCALARS:raise _BudgetCopyFallback
            result[key]=child if type(child) in _BUDGET_COPY_SCALARS else _budget_json_clone(child,memo)
    return result

def _budget_copy(value):
    try:result=_budget_json_clone(value,{})
    except _BudgetCopyFallback:
        _BUDGET_COPY_STATS['fallback']+=1
        return copy.deepcopy(value)
    _BUDGET_COPY_STATS['fast']+=1
    return result


# Public aliases; the helper above remains byte-for-byte unchanged.
clone_state = _budget_copy
COPY_STATS = _BUDGET_COPY_STATS
