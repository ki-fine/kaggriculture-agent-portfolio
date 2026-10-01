"""Leakage-free observation and complete-action schema for V64 behavior cloning.

This module intentionally depends only on the standard library and NumPy.  It
does not import a model or initialize an accelerator.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
MAX_ACTORS = 32
MAX_ORDERS = 10
UNIT_PAD_ID = -100
MARKET_PAD_ID = -100
MAX_SHOPS = 8

PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
CROPS = PRODUCTS[:5]
ANIMALS = ("COW", "SHEEP", "GOOSE")
INVENTORY_ITEMS = PRODUCTS + ANIMALS
SHOPS = (
    "BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
    "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE",
)
QUADRANTS = ("NW", "NE", "SW", "SE")

_TILE_KINDS = ("EMPTY", "LOCKED", "WEED", "PLANT", "COOP", "PASTURE", "UNKNOWN")
_TILE_CROPS = ("NONE",) + CROPS + ("UNKNOWN",)
_TILE_ANIMALS = ("NONE",) + ANIMALS + ("UNKNOWN",)
_SHOP_TYPES = SHOPS + ("UNKNOWN",)


def _tile_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    names.extend(f"kind.{value}" for value in _TILE_KINDS)
    names.extend(f"crop.{value}" for value in _TILE_CROPS)
    names.extend(f"animal.{value}" for value in _TILE_ANIMALS)
    names.extend((
        "planted.present", "planted.day", "planted.age",
        "placed.present", "placed.day", "placed.age",
        "lifespan.present", "lifespan.max_step", "lifespan.remaining",
        "yield_units", "watered_today", "consecutive_unwatered",
        "fed_today", "consecutive_unfed", "cared_today", "pending_care_bonus",
        "fertilizer_available", "fertilized_until.present",
        "fertilized_until.day", "fertilized_until.remaining",
    ))
    return tuple(names)


TILE_FEATURE_NAMES = _tile_feature_names()
TILE_DIM = len(TILE_FEATURE_NAMES)


def _public_actor_names(prefix: str) -> list[str]:
    result: list[str] = []
    for actor in range(MAX_ACTORS):
        result.extend((
            f"{prefix}.actor.{actor}.exists", f"{prefix}.actor.{actor}.is_farmer",
            f"{prefix}.actor.{actor}.index", f"{prefix}.actor.{actor}.x",
            f"{prefix}.actor.{actor}.y",
        ))
    return result


def _global_feature_names() -> tuple[str, ...]:
    names = [
        "time.step", "time.day", "time.hour", "time.remaining_overage",
        "player.0", "player.1", "history.present",
    ]
    names.extend(f"time.day_onehot.{day}" for day in range(30))
    names.extend(f"time.hour_onehot.{hour}" for hour in range(24))
    for side in ("self", "opponent"):
        names.extend((f"{side}.money", f"{side}.hires_today", f"{side}.hands_count"))
        names.extend(f"{side}.quadrant.{quadrant}" for quadrant in QUADRANTS)
        names.extend(_public_actor_names(side))
        for y in range(10):
            for x in range(10):
                names.extend(f"{side}.tile.{y}.{x}.{name}" for name in TILE_FEATURE_NAMES)
    names.extend(f"private.shed.{item}" for item in INVENTORY_ITEMS)
    names.extend(f"private.seeds.{crop}" for crop in CROPS)
    names.extend(f"market.price.{product}" for product in PRODUCTS)
    names.extend(f"market.inventory.{product}" for product in PRODUCTS)
    names.append("town.shop_count")
    for slot in range(MAX_SHOPS):
        names.append(f"town.shop.{slot}.exists")
        names.extend(f"town.shop.{slot}.{shop}" for shop in _SHOP_TYPES)
    names.extend(("delta.self_money", "delta.opponent_money"))
    names.extend(f"delta.market.price.{product}" for product in PRODUCTS)
    names.extend(f"delta.market.inventory.{product}" for product in PRODUCTS)
    return tuple(names)


def _actor_feature_names() -> tuple[str, ...]:
    names = ["exists", "is_farmer", "index", "x", "y"]
    names.extend(f"held.{item}" for item in INVENTORY_ITEMS)
    names.extend(f"tile.{name}" for name in TILE_FEATURE_NAMES)
    return tuple(names)


GLOBAL_FEATURE_NAMES = _global_feature_names()
ACTOR_FEATURE_NAMES = _actor_feature_names()
GLOBAL_DIM = len(GLOBAL_FEATURE_NAMES)
ACTOR_DIM = len(ACTOR_FEATURE_NAMES)

FEATURE_SCHEMA = {
    "schema_version": SCHEMA_VERSION,
    "max_actors": MAX_ACTORS,
    "max_orders": MAX_ORDERS,
    "global_dim": GLOBAL_DIM,
    "actor_dim": ACTOR_DIM,
    "tile_dim": TILE_DIM,
    "global_feature_names": GLOBAL_FEATURE_NAMES,
    "actor_feature_names": ACTOR_FEATURE_NAMES,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else ()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _count(value: Any, scale: float = 8.0) -> float:
    return math.tanh(math.log1p(max(0.0, _finite(value))) / scale)


def _signed(value: Any, scale: float) -> float:
    return math.tanh(_finite(value) / scale)


def _flag(value: Any) -> float:
    return float(bool(value))


def _one_hot(value: str, vocabulary: Sequence[str]) -> list[float]:
    return [float(value == choice) for choice in vocabulary]


def _tile_features(tile: Any, day: int, step: int) -> list[float]:
    if tile is None:
        kind, data = "EMPTY", {}
    elif tile == "LOCKED":
        kind, data = "LOCKED", {}
    elif isinstance(tile, Mapping):
        raw_kind = str(tile.get("kind", "")).upper()
        kind = raw_kind if raw_kind in _TILE_KINDS[:-1] else "UNKNOWN"
        data = tile
    else:
        kind, data = "UNKNOWN", {}

    crop = str(data.get("crop") or "NONE").upper()
    if crop not in _TILE_CROPS[:-1]:
        crop = "UNKNOWN"
    animal = str(data.get("animal") or "NONE").upper()
    if animal not in _TILE_ANIMALS[:-1]:
        animal = "UNKNOWN"
    values = _one_hot(kind, _TILE_KINDS)
    values.extend(_one_hot(crop, _TILE_CROPS))
    values.extend(_one_hot(animal, _TILE_ANIMALS))

    planted_present = "planted_day" in data
    planted_day = int(_finite(data.get("planted_day"), 0.0))
    placed_present = "placed_day" in data
    placed_day = int(_finite(data.get("placed_day"), 0.0))
    lifespan = int(_finite(data.get("max_lifespan_step"), -1.0))
    lifespan_present = lifespan >= 0
    fertilized_until = int(_finite(data.get("fertilized_until_day"), -1.0))
    fertilized_present = fertilized_until >= 0
    values.extend((
        float(planted_present), _signed(planted_day, 29.0) if planted_present else 0.0,
        _signed(day - planted_day, 29.0) if planted_present else 0.0,
        float(placed_present), _signed(placed_day, 29.0) if placed_present else 0.0,
        _signed(day - placed_day, 29.0) if placed_present else 0.0,
        float(lifespan_present), _signed(lifespan, 719.0) if lifespan_present else 0.0,
        _signed(lifespan - step, 719.0) if lifespan_present else 0.0,
        _count(data.get("yield_units"), 4.0),
        _flag(data.get("watered_today")), _count(data.get("consecutive_unwatered"), 3.0),
        _flag(data.get("fed_today")), _count(data.get("consecutive_unfed"), 3.0),
        _flag(data.get("cared_today")), _count(data.get("pending_care_bonus"), 3.0),
        _flag(data.get("fertilizer_available")), float(fertilized_present),
        _signed(fertilized_until, 29.0) if fertilized_present else 0.0,
        _signed(fertilized_until - day, 29.0) if fertilized_present else 0.0,
    ))
    if len(values) != TILE_DIM:
        raise AssertionError("tile feature dimension drift")
    return values


def _farm(observation: Mapping[str, Any], player: int) -> Mapping[str, Any]:
    farms = _sequence(observation.get("farms"))
    if len(farms) != 2 or not isinstance(farms[player], Mapping):
        raise ValueError("observation must contain two public farms")
    return farms[player]


def _tiles(farm: Mapping[str, Any]) -> Sequence[Sequence[Any]]:
    tiles = _sequence(farm.get("tiles"))
    if len(tiles) != 10 or any(len(_sequence(row)) != 10 for row in tiles):
        raise ValueError("farm tiles must be a complete 10x10 board")
    return tiles  # type: ignore[return-value]


def _positions(farm: Mapping[str, Any]) -> list[Sequence[Any]]:
    farmer = _sequence(farm.get("farmer"))
    hands = _sequence(farm.get("hands"))
    positions = [farmer, *hands]
    if len(positions) > MAX_ACTORS:
        raise ValueError(f"actor count exceeds MAX_ACTORS={MAX_ACTORS}")
    for position in positions:
        if len(position) != 2:
            raise ValueError("actor position must contain x and y")
        x, y = int(_finite(position[0], -1)), int(_finite(position[1], -1))
        if not 0 <= x < 10 or not 0 <= y < 10:
            raise ValueError("actor position is outside the 10x10 board")
    return positions


def _public_farm_features(farm: Mapping[str, Any], day: int, step: int) -> list[float]:
    hands = _sequence(farm.get("hands"))
    values = [
        _signed(math.log1p(max(0.0, _finite(farm.get("money")))), 12.0),
        _count(farm.get("hires_today"), 4.0),
        _count(len(hands), 4.0),
    ]
    unlocked = {str(value).upper() for value in _sequence(farm.get("unlocked_quadrants"))}
    values.extend(float(quadrant in unlocked) for quadrant in QUADRANTS)
    positions = _positions(farm)
    for actor in range(MAX_ACTORS):
        if actor < len(positions):
            x, y = (int(positions[actor][0]), int(positions[actor][1]))
            values.extend((1.0, float(actor == 0), actor / (MAX_ACTORS - 1), x / 9.0, y / 9.0))
        else:
            values.extend((0.0, 0.0, actor / (MAX_ACTORS - 1), 0.0, 0.0))
    for row in _tiles(farm):
        for tile in row:
            values.extend(_tile_features(tile, day, step))
    return values


def encode_observation(
    observation: Mapping[str, Any], previous: Mapping[str, Any] | None = None
) -> dict[str, np.ndarray]:
    """Encode one seat's visible state without route, future, or enemy-private data."""
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be a mapping")
    player = int(_finite(observation.get("player"), -1.0))
    if player not in (0, 1):
        raise ValueError("player must be seat 0 or 1")
    step = int(_finite(observation.get("step"), 0.0))
    day = int(_finite(observation.get("day"), step // 24))
    hour = int(_finite(observation.get("hour"), step % 24))
    previous_ok = isinstance(previous, Mapping) and int(_finite(previous.get("player"), -1)) == player

    own = _farm(observation, player)
    opponent = _farm(observation, 1 - player)
    values: list[float] = [
        _signed(step, 719.0), _signed(day, 29.0), _signed(hour, 23.0),
        _signed(observation.get("remainingOverageTime"), 60.0),
        float(player == 0), float(player == 1), float(previous_ok),
    ]
    values.extend(float(day == value) for value in range(30))
    values.extend(float(hour == value) for value in range(24))
    values.extend(_public_farm_features(own, day, step))
    values.extend(_public_farm_features(opponent, day, step))

    private = _mapping(observation.get("private"))
    shed = _mapping(private.get("shed"))
    seeds = _mapping(private.get("seeds"))
    values.extend(_count(shed.get(item)) for item in INVENTORY_ITEMS)
    values.extend(_count(seeds.get(crop)) for crop in CROPS)

    market = _mapping(observation.get("market"))
    prices = _mapping(market.get("prices"))
    inventory = _mapping(market.get("inventory"))
    values.extend(_count(prices.get(product), 8.0) for product in PRODUCTS)
    values.extend(_count(inventory.get(product), 10.0) for product in PRODUCTS)

    shops = _sequence(_mapping(observation.get("town")).get("unlocked_shops"))
    if len(shops) > MAX_SHOPS:
        raise ValueError(f"shop history exceeds MAX_SHOPS={MAX_SHOPS}")
    values.append(len(shops) / MAX_SHOPS)
    for slot in range(MAX_SHOPS):
        if slot < len(shops):
            shop = str(shops[slot]).upper()
            if shop not in SHOPS:
                shop = "UNKNOWN"
            values.append(1.0)
            values.extend(_one_hot(shop, _SHOP_TYPES))
        else:
            values.append(0.0)
            values.extend(0.0 for _ in _SHOP_TYPES)

    if previous_ok:
        assert previous is not None
        previous_own = _farm(previous, player)
        previous_opponent = _farm(previous, 1 - player)
        previous_market = _mapping(previous.get("market"))
        previous_prices = _mapping(previous_market.get("prices"))
        previous_inventory = _mapping(previous_market.get("inventory"))
        values.extend((
            _signed(_finite(own.get("money")) - _finite(previous_own.get("money")), 5000.0),
            _signed(_finite(opponent.get("money")) - _finite(previous_opponent.get("money")), 5000.0),
        ))
        values.extend(_signed(_finite(prices.get(product)) - _finite(previous_prices.get(product)), 100.0)
                      for product in PRODUCTS)
        values.extend(_signed(_finite(inventory.get(product)) - _finite(previous_inventory.get(product)), 1000.0)
                      for product in PRODUCTS)
    else:
        values.extend(0.0 for _ in range(2 + 2 * len(PRODUCTS)))

    if len(values) != GLOBAL_DIM:
        raise AssertionError(f"global feature dimension drift: {len(values)} != {GLOBAL_DIM}")

    positions = _positions(own)
    inventories = _sequence(private.get("inventories"))
    if len(inventories) > MAX_ACTORS:
        raise ValueError(f"private inventories exceed MAX_ACTORS={MAX_ACTORS}")
    own_tiles = _tiles(own)
    actors = np.zeros((MAX_ACTORS, ACTOR_DIM), dtype=np.float32)
    for actor, position in enumerate(positions):
        x, y = int(position[0]), int(position[1])
        held = _mapping(inventories[actor]) if actor < len(inventories) else {}
        row = [1.0, float(actor == 0), actor / (MAX_ACTORS - 1), x / 9.0, y / 9.0]
        row.extend(_count(held.get(item)) for item in INVENTORY_ITEMS)
        row.extend(_tile_features(own_tiles[y][x], day, step))
        actors[actor] = np.asarray(row, dtype=np.float32)

    return {"global": np.asarray(values, dtype=np.float32), "actors": actors}


_UNIT_NO_ARG = {
    "NORTH", "SOUTH", "EAST", "WEST", "PASS", "DROP", "WATER", "HARVEST",
    "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "COLLECT_FERTILIZER",
    "CARE", "DIG",
}
_UNIT_ITEM = {"PICKUP", "PLACE"}
_MARKET_NO_ARG = {"HIRE", "BUY_LAND"}
_MARKET_QUANTITY = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL"}


def _command_sequence(command: Any) -> tuple[Any, ...] | None:
    if not isinstance(command, Sequence) or isinstance(command, (str, bytes)):
        return None
    return tuple(command)


def _quantity(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and int(value) >= 0


def _valid_unit(command: tuple[Any, ...]) -> bool:
    if not command or not isinstance(command[0], str):
        return False
    op = command[0]
    if op in _UNIT_NO_ARG:
        return len(command) == 1
    if op == "PLANT":
        return len(command) == 2 and command[1] in CROPS
    if op in _UNIT_ITEM:
        return (len(command) in (2, 3) and command[1] in INVENTORY_ITEMS and
                (len(command) == 2 or _quantity(command[2])))
    return False


def _valid_market(command: tuple[Any, ...]) -> bool:
    if command == ():
        return True
    if not command or not isinstance(command[0], str):
        return False
    op = command[0]
    if op in _MARKET_NO_ARG:
        return len(command) == 1
    if op not in _MARKET_QUANTITY or len(command) != 3 or not _quantity(command[2]):
        return False
    item = command[1]
    if op == "BUY_SEED":
        return item in CROPS
    if op == "BUY_ANIMAL":
        return item in ANIMALS
    return item in PRODUCTS


def _sort_key(command: tuple[Any, ...]) -> str:
    return json.dumps(list(command), ensure_ascii=False, separators=(",", ":"))


def _parts(action: Mapping[str, Any]) -> tuple[tuple[Any, ...], list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    if not isinstance(action, Mapping):
        raise ValueError("action must be a mapping")
    farmer = _command_sequence(action.get("farmer", ["PASS"]))
    hands_raw = action.get("hands", [])
    market_raw = action.get("market", [])
    if farmer is None or not isinstance(hands_raw, list) or not isinstance(market_raw, list):
        raise ValueError("action farmer/hands/market have invalid container types")
    hands = [_command_sequence(command) for command in hands_raw]
    market = [_command_sequence(command) for command in market_raw]
    if any(command is None for command in hands + market):
        raise ValueError("commands must be list-like sequences")
    if len(hands) + 1 > MAX_ACTORS:
        raise ValueError(f"action actor count exceeds MAX_ACTORS={MAX_ACTORS}")
    return farmer, hands, market  # type: ignore[return-value]


class ActionCodec:
    """Deterministic training-only vocabularies for complete unit/market actions."""

    def __init__(self, unit_commands: Sequence[tuple[Any, ...] | None],
                 market_commands: Sequence[tuple[Any, ...] | None]) -> None:
        self._unit_commands = tuple(unit_commands)
        self._market_commands = tuple(market_commands)
        self.unit_unk_id = 0
        self.unit_pass_id = 1
        self.market_unk_id = 0
        self.market_bos_id = 1
        self.market_eos_id = 2
        if len(self._unit_commands) < 2 or self._unit_commands[:2] != (None, ("PASS",)):
            raise ValueError("unit vocabulary must start with UNK, PASS")
        if len(self._market_commands) < 3 or self._market_commands[:3] != (None, None, None):
            raise ValueError("market vocabulary must start with UNK, BOS, EOS")
        if any(command is None for command in self._unit_commands[2:]):
            raise ValueError("unit command vocabulary contains a reserved token")
        if any(command is None for command in self._market_commands[3:]):
            raise ValueError("market command vocabulary contains a reserved token")
        self._unit_to_id = {command: index for index, command in enumerate(self._unit_commands) if command is not None}
        self._market_to_id = {command: index for index, command in enumerate(self._market_commands) if command is not None}
        if len(self._unit_to_id) != len(self._unit_commands) - 1:
            raise ValueError("duplicate unit commands")
        if len(self._market_to_id) != len(self._market_commands) - 3:
            raise ValueError("duplicate market commands")
        self.audit = {
            "unit_unknown": 0, "market_unknown": 0,
            "market_truncated_orders": 0, "decode_unknown": 0,
        }
        self.last_encode_report: dict[str, Any] = {}
        self.last_decode_report: dict[str, Any] = {}

    @classmethod
    def fit(cls, actions: Iterable[Mapping[str, Any]]) -> "ActionCodec":
        units: set[tuple[Any, ...]] = {("PASS",)}
        markets: set[tuple[Any, ...]] = set()
        for action_index, action in enumerate(actions):
            farmer, hands, market = _parts(action)
            for command in (farmer, *hands):
                if not _valid_unit(command):
                    raise ValueError(f"non-standard unit command at action {action_index}: {command!r}")
                units.add(command)
            for command in market:
                if not _valid_market(command):
                    raise ValueError(f"non-standard market command at action {action_index}: {command!r}")
            markets.update(market[:MAX_ORDERS])
        unit_commands: list[tuple[Any, ...] | None] = [None, ("PASS",)]
        unit_commands.extend(sorted(units - {("PASS",)}, key=_sort_key))
        market_commands: list[tuple[Any, ...] | None] = [None, None, None]
        market_commands.extend(sorted(markets, key=_sort_key))
        return cls(unit_commands, market_commands)

    @property
    def unit_size(self) -> int:
        return len(self._unit_commands)

    @property
    def market_size(self) -> int:
        return len(self._market_commands)

    @property
    def unit_vocab_size(self) -> int:
        return self.unit_size

    @property
    def market_vocab_size(self) -> int:
        return self.market_size

    def encode(self, action: Mapping[str, Any]) -> dict[str, np.ndarray]:
        farmer, hands, market = _parts(action)
        unit_ids = np.full(MAX_ACTORS, UNIT_PAD_ID, dtype=np.int64)
        market_ids = np.full(MAX_ORDERS + 1, MARKET_PAD_ID, dtype=np.int64)
        unit_unknown: list[dict[str, Any]] = []
        market_unknown: list[dict[str, Any]] = []
        for actor, command in enumerate((farmer, *hands)):
            token = self._unit_to_id.get(command) if _valid_unit(command) else None
            if token is None:
                token = self.unit_unk_id
                unit_unknown.append({"actor": actor, "command": list(command)})
            unit_ids[actor] = token
        truncated = max(0, len(market) - MAX_ORDERS)
        for index, command in enumerate(market[:MAX_ORDERS]):
            token = self._market_to_id.get(command) if _valid_market(command) else None
            if token is None:
                token = self.market_unk_id
                market_unknown.append({"index": index, "command": list(command)})
            market_ids[index] = token
        market_ids[min(len(market), MAX_ORDERS)] = self.market_eos_id
        self.audit["unit_unknown"] += len(unit_unknown)
        self.audit["market_unknown"] += len(market_unknown)
        self.audit["market_truncated_orders"] += truncated
        self.last_encode_report = {
            "unit_unknown": unit_unknown, "market_unknown": market_unknown,
            "market_truncated_orders": truncated,
            "truncated_market_commands": [list(command) for command in market[MAX_ORDERS:]],
            "market_overflow_rule": "truncate_tail_officially_ignored",
        }
        return {"units": unit_ids, "hands_length": np.int64(len(hands)), "market": market_ids}

    def decode(self, unit_ids: Sequence[int], hands_length: int,
               market_ids: Sequence[int]) -> dict[str, Any]:
        units = np.asarray(unit_ids).reshape(-1)
        markets = np.asarray(market_ids).reshape(-1)
        if units.shape != (MAX_ACTORS,):
            raise ValueError(f"unit_ids must have shape ({MAX_ACTORS},)")
        if markets.shape != (MAX_ORDERS + 1,):
            raise ValueError(f"market_ids must have shape ({MAX_ORDERS + 1},)")
        count = int(hands_length)
        if not 0 <= count < MAX_ACTORS:
            raise ValueError("hands_length is outside the actor contract")
        unit_unknown: list[int] = []
        decoded_units: list[list[Any]] = []
        for actor in range(count + 1):
            token = int(units[actor])
            command = self._unit_commands[token] if 0 <= token < self.unit_size else None
            if command is None:
                command = ("PASS",)
                unit_unknown.append(actor)
            decoded_units.append(list(command))

        orders: list[list[Any]] = []
        market_unknown: list[int] = []
        saw_eos = False
        for index, raw_token in enumerate(markets):
            token = int(raw_token)
            if token == self.market_eos_id:
                saw_eos = True
                break
            if len(orders) >= MAX_ORDERS:
                break
            command = self._market_commands[token] if 0 <= token < self.market_size else None
            if command is None:
                command = ()
                market_unknown.append(index)
            orders.append(list(command))
        unknown_count = len(unit_unknown) + len(market_unknown) + int(not saw_eos)
        self.audit["decode_unknown"] += unknown_count
        self.last_decode_report = {
            "unit_unknown_actors": unit_unknown,
            "market_unknown_indices": market_unknown,
            "missing_market_eos": not saw_eos,
            "safe_unit_fallback": "PASS",
            "safe_market_fallback": [],
        }
        return {"farmer": decoded_units[0], "hands": decoded_units[1:], "market": orders}

    def to_dict(self) -> dict[str, Any]:
        def serialise(commands: Sequence[tuple[Any, ...] | None]) -> list[Any]:
            return [None if command is None else list(command) for command in commands]
        return {
            "schema_version": SCHEMA_VERSION,
            "max_actors": MAX_ACTORS,
            "max_orders": MAX_ORDERS,
            "unit_commands": serialise(self._unit_commands),
            "market_commands": serialise(self._market_commands),
            "unit_unk_id": self.unit_unk_id,
            "unit_pass_id": self.unit_pass_id,
            "market_unk_id": self.market_unk_id,
            "market_bos_id": self.market_bos_id,
            "market_eos_id": self.market_eos_id,
            "market_overflow_rule": "truncate_tail_officially_ignored",
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "ActionCodec":
        if int(document.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("unsupported action codec schema version")
        if int(document.get("max_actors", -1)) != MAX_ACTORS or int(document.get("max_orders", -1)) != MAX_ORDERS:
            raise ValueError("action codec dimensions do not match this schema")
        unit_raw = document.get("unit_commands")
        market_raw = document.get("market_commands")
        if not isinstance(unit_raw, list) or not isinstance(market_raw, list):
            raise ValueError("action codec vocabularies must be lists")
        units = [None if command is None else _command_sequence(command) for command in unit_raw]
        markets = [None if command is None else _command_sequence(command) for command in market_raw]
        if any(command is None and raw is not None for command, raw in zip(units, unit_raw)):
            raise ValueError("invalid unit command serialization")
        if any(command is None and raw is not None for command, raw in zip(markets, market_raw)):
            raise ValueError("invalid market command serialization")
        codec = cls(units, markets)
        if any(not _valid_unit(command) for command in codec._unit_commands[1:] if command is not None):
            raise ValueError("serialized unit vocabulary contains a non-standard command")
        if any(not _valid_market(command) for command in codec._market_commands[3:] if command is not None):
            raise ValueError("serialized market vocabulary contains a non-standard command")
        expected = {
            "unit_unk_id": codec.unit_unk_id, "unit_pass_id": codec.unit_pass_id,
            "market_unk_id": codec.market_unk_id, "market_bos_id": codec.market_bos_id,
            "market_eos_id": codec.market_eos_id,
        }
        if any(int(document.get(key, -1)) != value for key, value in expected.items()):
            raise ValueError("reserved action token ids changed")
        if document.get("market_overflow_rule") != "truncate_tail_officially_ignored":
            raise ValueError("unknown market overflow rule")
        return codec


assert TILE_DIM == 39
assert GLOBAL_DIM == 8331
assert ACTOR_DIM == 56
