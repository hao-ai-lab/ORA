"""The pinned BDDL language of the forty source-paired LIBERO-Pro tasks.

LIBERO-Pro perturbs each of the forty tasks of LIBERO-Spatial, -Object, -Goal
and LIBERO-10 four ways (``_task``, ``_swap``, ``_lan``, ``_object``; see
``libero_pro.CORE_VARIANT_SUFFIXES``). Each row is one task in the benchmark's
order with the ``(:language ...)`` field of its four core BDDL files at
``LIBERO_PRO_DATA_REVISION``. That field is the instruction a core-cell
RunSpec delivers (``runtime.expected_task_language``) and launch preflight
compares it with the BDDL it loads (``orchestrator/libero_preflight.py``), so a
row that drifts from the data is refused before a run, never delivered. A base
cell delivers the official file-name language instead (``base_task_language``).

The rows moved here from the Viewer's presentation catalog
(``viewer/libero_pro_catalog.py``, which imports them) on 2026-10-05 so a
maintained RunSpec builder reads them from ``ora``; the text is
unchanged.
"""

from __future__ import annotations

from .libero_pro import (
    BASE_SUITES, CORE_SUITES, CORE_VARIANT_SUFFIXES, FAMILY_ORDER, LiberoProBenchmarkError, base_suite,
)

#: family, task_id, canonical filename, then the task/swap/lan/object BDDL language.
#: Order is the benchmark's task order, not lexical filename order.
TASK_ROWS = (
    ("object", 0, "pick_up_the_alphabet_soup_and_place_it_in_the_basket", "Pick the cream cheese and place it in the basket", "Pick the alphabet soup and place it in the basket", "grab alphabet soup and put it into basket", "Pick the alphabet soup and place it in the basket"),
    ("object", 1, "pick_up_the_cream_cheese_and_place_it_in_the_basket", "Pick the alphabet soup and place it in the basket", "Pick the cream cheese and place it in the basket", "grab cream cheese and put into basket", "Pick the cream cheese and place it in the basket"),
    ("object", 2, "pick_up_the_salad_dressing_and_place_it_in_the_basket", "Pick the tomato sauce and place it in the basket", "Pick the salad dressing and place it in the basket", "grab salad dressing and put into basket", "Pick the salad dressing and place it in the basket"),
    ("object", 3, "pick_up_the_bbq_sauce_and_place_it_in_the_basket", "Pick the ketchup and place it in the basket", "Pick the bbq sauce and place it in the basket", "pick up bbq sauce and put it into basket", "Pick the bbq sauce and place it in the basket"),
    ("object", 4, "pick_up_the_ketchup_and_place_it_in_the_basket", "Pick the milk and place it in the basket", "Pick the ketchup and place it in the basket", "lift ketchup and put it in basket", "Pick the ketchup and place it in the basket"),
    ("object", 5, "pick_up_the_tomato_sauce_and_place_it_in_the_basket", "Pick the bbq sauce and place it in the basket", "Pick the tomato sauce and place it in the basket", "pick tomato sauce and put it in basket", "Pick the tomato sauce and place it in the basket"),
    ("object", 6, "pick_up_the_butter_and_place_it_in_the_basket", "Pick the orange juice and place it in the basket", "Pick the butter and place it in the basket", "grab butter and put it into basket", "Pick the butter and place it in the basket"),
    ("object", 7, "pick_up_the_milk_and_place_it_in_the_basket", "Pick the butter and place it in the basket", "Pick the milk and place it in the basket", "grab milk and put into basket", "Pick the milk and place it in the basket"),
    ("object", 8, "pick_up_the_chocolate_pudding_and_place_it_in_the_basket", "Pick the salad dressing and place it in the basket", "Pick the chocolate pudding and place it in the basket", "lift chocolate pudding and put it in basket", "Pick the chocolate pudding and place it in the basket"),
    ("object", 9, "pick_up_the_orange_juice_and_place_it_in_the_basket", "Pick the chocolate pudding and place it in the basket", "Pick the orange juice and place it in the basket", "pick orange juice and put into basket", "Pick the orange juice and place it in the basket"),
    ("goal", 0, "open_the_middle_drawer_of_the_cabinet", "open the bottom drawer of the cabinet", "Open the middle layer of the drawer", "open the cabinet’s middle drawer", "Open the middle layer of the drawer"),
    ("goal", 1, "put_the_bowl_on_the_stove", "Put the plate on the stove", "Put the bowl on the stove", "place the bowl on the stove", "Put the bowl on the stove"),
    ("goal", 2, "put_the_wine_bottle_on_top_of_the_cabinet", "put the wine bottle in the bowl", "Put the wine bottle on the top of the drawer", "place the wine bottle on top of the cabinet", "Put the wine bottle on the top of the drawer"),
    ("goal", 3, "open_the_top_drawer_and_put_the_bowl_inside", "Open the top layer of the drawer and put the cream cheese inside", "Open the top layer of the drawer and put the bowl inside", "slide open the top drawer and place the bowl in it", "Open the top layer of the drawer and put the bowl inside"),
    ("goal", 4, "put_the_bowl_on_top_of_the_cabinet", "Put the plate on the top of the drawer", "Put the bowl on the top of the drawer", "place the bowl on top of the cabinet", "Put the bowl on the top of the drawer"),
    ("goal", 5, "push_the_plate_to_the_front_of_the_stove", "Push the cream cheese to the front of the stove", "Push the plate to the front of the stove", "move the plate to the stove’s front", "Push the plate to the front of the stove"),
    ("goal", 6, "put_the_cream_cheese_in_the_bowl", "put the wine bottle in the bowl", "Put the cream cheese on the bowl", "place the cream cheese inside the bowl", "Put the cream cheese on the bowl"),
    ("goal", 7, "turn_on_the_stove", "Turn off the stove", "Turn on the stove", "switch on the stove", "Turn on the stove"),
    ("goal", 8, "put_the_bowl_on_the_plate", "Put the wine bottle on the plate", "Put the bowl on the plate", "place the bowl on top of the plate", "Put the bowl on the plate"),
    ("goal", 9, "put_the_wine_bottle_on_the_rack", "Put the cream cheese on the rack", "Put the wine bottle on the rack", "place the wine bottle on the rack", "Put the wine bottle on the rack"),
    ("spatial", 0, "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate", "Pick the akita black bowl not between the plate and the ramekin and place it on the plate", "Pick the akita black bowl between the plate and the ramekin and place it on the plate", "lift the black bowl between the plate and ramekin and set it on the plate", "Pick the akita black bowl between the plate and the ramekin and place it on the plate"),
    ("spatial", 1, "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate", "Pick the akita black bowl next to the cookie box and place it on the plate", "Pick the akita black bowl next to the ramekin and place it on the plate", "take black bowl next to ramekin and place on plate", "Pick the akita black bowl next to the ramekin and place it on the plate"),
    ("spatial", 2, "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate", "Pick the akita black bowl next to the plate and place it on the plate", "Pick the akita black bowl from table center and place it on the plate", "lift the black bowl from the table center and put it on the plate", "Pick the akita black bowl from table center and place it on the plate"),
    ("spatial", 3, "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate", "Pick the akita black bowl on the top of the cabinet and place it on the plate", "Pick the akita black bowl on the cookies box and place it on the plate", "lift black bowl from cookie box and put it on plate", "Pick the akita black bowl on the cookies box and place it on the plate"),
    ("spatial", 4, "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate", "Pick the akita black bowl on the top of the wooden cabinet and place it on the plate", "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate", "take the black bowl from the top drawer and put it on plate", "Pick the akita black bowl in the top layer of the wooden cabinet and place it on the plate"),
    ("spatial", 5, "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate", "Pick the akita black bowl on the cookie box and place it on the plate", "Pick the akita black bowl on the ramekin and place it on the plate", "grab black bowl on ramekin and set it on plate", "Pick the akita black bowl on the ramekin and place it on the plate"),
    ("spatial", 6, "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate", "Pick the akita black bowl on the stove and place it on the plate", "Pick the akita black bowl next to the cookies box and place it on the plate", "grab the black bowl beside the cookie box and set it on plate", "Pick the akita black bowl next to the cookies box and place it on the plate"),
    ("spatial", 7, "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate", "Pick the akita black bowl on the top of the cabinet and place it on the plate", "Pick the akita black bowl on the stove and place it on the plate", "lift bowl from stove and set on plate", "Pick the akita black bowl on the stove and place it on the plate"),
    ("spatial", 8, "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate", "Pick the akita black bowl next to the ramekin and place it on the plate", "Pick the akita black bowl next to the plate and place it on the plate", "pick the bowl beside the plate and put it on plate", "Pick the akita black bowl next to the plate and place it on the plate"),
    ("spatial", 9, "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate", "Pick the akita black bowl on the stove and place it on the plate", "Pick the akita black bowl on the wooden cabinet and place it on the plate", "lift bowl from cabinet and put on plate", "Pick the akita black bowl on the wooden cabinet and place it on the plate"),
    ("libero_10", 0, "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", "put both the cream cheese and the tomato sauce in the basket", "put both the alphabet soup and the tomato sauce in the basket", "place alphabet soup and tomato sauce into basket", "put both the alphabet soup and the tomato sauce in the basket"),
    ("libero_10", 1, "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket", "put both the alphabet soup and the butter in the basket", "put both the cream cheese box and the butter in the basket", "place cream cheese box and butter into basket", "put both the cream cheese box and the butter in the basket"),
    ("libero_10", 2, "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it", "turn on the stove and put the pan on it", "turn on the stove and put the moka pot on it", "switch stove on and put moka pot on it", "turn on the stove and put the moka pot on it"),
    ("libero_10", 3, "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it", "put the bottle in the bottom drawer of the cabinet and close it", "put the black bowl in the bottom drawer of the cabinet and close it", "put black bowl inside bottom drawer and close it", "put the black bowl in the bottom drawer of the cabinet and close it"),
    ("libero_10", 4, "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate", "put the yellow and white mug on the left plate and put the white mug on the right plate", "put the white mug on the left plate and put the yellow and white mug on the right plate", "place white mug on left plate and yellow-white mug on right plate", "put the white mug on the left plate and put the yellow and white mug on the right plate"),
    ("libero_10", 5, "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy", "pick up the cup and place it in the back compartment of the caddy", "pick up the book and place it in the back compartment of the caddy", "pick up the book and place in back compartment", "pick up the book and place it in the back compartment of the caddy"),
    ("libero_10", 6, "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate", "put the red mug on the plate and put the chocolate pudding to the right of the plate", "put the white mug on the plate and put the chocolate pudding to the right of the plate", "place white mug on plate and chocolate pudding to right", "put the white mug on the plate and put the chocolate pudding to the right of the plate"),
    ("libero_10", 7, "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket", "put both the ketchup and the cream cheese box in the basket", "put both the alphabet soup and the cream cheese box in the basket", "put alphabet soup and cream cheese box in basket", "put both the alphabet soup and the cream cheese box in the basket"),
    ("libero_10", 8, "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove", "put the left moka pot on the stove", "put both moka pots on the stove", "place both moka pots on stove", "put both moka pots on the stove"),
    ("libero_10", 9, "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it", "put the white mug in the microwave and close it", "put the yellow and white mug in the microwave and close it", "place yellow and white mug in microwave and close door", "put the yellow and white mug in the microwave and close it"),
)
#: The row field that holds each core suffix's language.
_LANGUAGE_FIELD = {"task": 3, "swap": 4, "lan": 5, "object": 6}
_ROWS = {(row[0], row[1]): row for row in TASK_ROWS}
_SUITE_FAMILY_SUFFIX = {
    f"{base_suite(family)}_{suffix}": (family, suffix)
    for family in FAMILY_ORDER for suffix in CORE_VARIANT_SUFFIXES.values()
}


def core_task_language(suite: str, task_id: int) -> str:
    """The BDDL language of task ``task_id`` (0-9) of core suite ``suite``."""
    if suite not in CORE_SUITES:
        raise LiberoProBenchmarkError(f"not a LIBERO-Pro core suite: {suite!r}")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < 10:
        raise LiberoProBenchmarkError(f"{suite} task_id must be in [0,9]")
    family, suffix = _SUITE_FAMILY_SUFFIX[suite]
    return _ROWS[(family, task_id)][_LANGUAGE_FIELD[suffix]]


def language_from_filename(name: str) -> str:
    """LIBERO's ``Task.language``: the benchmark registry's rule for turning a BDDL file name into text.

    Mirrors ``libero.libero.benchmark.grab_language_from_filename``: a LIBERO-100
    scene file drops its ``<ROOM>_SCENE<n>_`` prefix, every other name only
    replaces underscores with spaces.
    """
    name = name.removesuffix(".bddl")
    if name[0].isupper():
        name = name[name.find("SCENE") + (8 if "SCENE10" in name else 7):]
    return " ".join(name.split("_"))


def _base_row(suite: str, task_id: int) -> tuple:
    if suite not in BASE_SUITES:
        raise LiberoProBenchmarkError(f"not a source-paired LIBERO base suite: {suite!r}")
    if isinstance(task_id, bool) or not isinstance(task_id, int) or not 0 <= task_id < 10:
        raise LiberoProBenchmarkError(f"{suite} task_id must be in [0,9]")
    family = next(family for family in FAMILY_ORDER if base_suite(family) == suite)
    return _ROWS[(family, task_id)]


def base_task_language(suite: str, task_id: int) -> str:
    """The official instruction of task ``task_id`` (0-9) of an original base suite (spatial, object, goal, 10).

    The original benchmark's evaluation feeds the policy ``Task.language``,
    which the registry derives from the BDDL file name, not the file's own
    ``(:language ...)`` field. For the base suites both describe the same
    objects and goal, but 30 of the 40 texts differ, 25 in wording (for example
    "pick up the black bowl in the top drawer" versus "Pick the akita black
    bowl in the top layer"), so a base cell instructs with the official text
    and records the BDDL text as a reviewed divergence
    (``base_bddl_language``). Perturbation suites keep the base file name while
    rewriting the goal, so core cells use ``core_task_language`` instead.
    """
    return language_from_filename(_base_row(suite, task_id)[2])


def base_bddl_language(suite: str, task_id: int) -> str:
    """The ``(:language ...)`` field of base task ``task_id``'s own BDDL file.

    A ``_swap`` variant moves objects and keeps its source task's goal and
    wording, so its column is the base BDDL language of all forty tasks.
    """
    return _base_row(suite, task_id)[_LANGUAGE_FIELD["swap"]]


if len(TASK_ROWS) != 40 or len(_ROWS) != 40 or set(_SUITE_FAMILY_SUFFIX) != set(CORE_SUITES):
    raise RuntimeError("the LIBERO-Pro core language table must cover forty tasks in sixteen suites")


__all__ = ["TASK_ROWS", "base_bddl_language", "base_task_language", "core_task_language", "language_from_filename"]
