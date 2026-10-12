"""Pinned RoboTwin task-gallery metadata migrated from Agent-as-Policy Atlas."""

from __future__ import annotations

from dataclasses import asdict, dataclass


CATALOG_SCHEMA = "ora.robotwin-task-catalog.v1"
ROBOTWIN_REVISION = "c3ddfa8b97d5519efa828b075999bd0006778e5e"
ROBOTWIN_DOC_REVISION = "f2c67e6f0b0dba88a5ff3a02f4937081abe578d2"
ROBOTWIN_MEDIA_ROOT = "https://robotwin-platform.github.io/doc/tasks/task_video_clean"
MIGRATED_FROM_COMMIT = "a762defefdecff6286ba3079cb02a77cd5327141"

EMBODIMENTS = (
    ("aloha-agilex", "Aloha-AgileX"),
    ("ARX-X5", "ARX-X5"),
    ("franka-panda", "Franka-Panda"),
    ("piper", "Piper"),
    ("ur5-wsg", "UR5-WSG"),
)
EMBODIMENT_IDS = tuple(item[0] for item in EMBODIMENTS)

FAMILY_LABELS = {
    "placement": ("搬运与空间关系", "Placement & spatial relations"),
    "bimanual": ("双臂协同", "Bimanual coordination"),
    "ordering": ("排序与堆叠", "Ordering & stacking"),
    "articulation": ("关节机构", "Articulated mechanisms"),
    "contact": ("精细接触与工具", "Precision contact & tools"),
    "reorientation": ("姿态与动态操作", "Reorientation & dynamics"),
    "affordance": ("特殊可供性", "Special affordances"),
}
FAMILY_ORDER = tuple(FAMILY_LABELS)

_TITLE_WORDS = {
    "a2b": "A→B",
    "alarmclock": "Alarm Clock",
    "bigbin": "Large Bin",
    "mic": "Microphone",
    "pillbottle": "Pill Bottle",
    "plasticbox": "Plastic Box",
    "qrcode": "QR Code",
    "rgb": "RGB",
}


def _title(task_id: str) -> str:
    return " ".join(
        _TITLE_WORDS.get(word, word[:1].upper() + word[1:])
        for word in task_id.split("_")
    )


@dataclass(frozen=True)
class RobotTwinTask:
    id: str
    title: str
    family: str
    requirement: str
    average_steps: int
    objects: tuple[str, ...]
    generation_success_percent: dict[str, int]
    video_embodiments: tuple[str, ...]
    official_page: str
    source_code: str

    def video_url(self, embodiment: str, viewpoint: str) -> str:
        if embodiment not in self.video_embodiments or viewpoint not in {"world", "head"}:
            raise ValueError("unsupported RoboTwin gallery video")
        return f"{ROBOTWIN_MEDIA_ROOT}/{self.id}/{embodiment}_{viewpoint}.mp4"


def _task(
    task_id: str,
    family: str,
    requirement: str,
    average_steps: int,
    objects: tuple[str, ...],
    rates: tuple[int, int, int, int, int],
    video_embodiments: tuple[str, ...] = EMBODIMENT_IDS,
) -> RobotTwinTask:
    return RobotTwinTask(
        id=task_id,
        title=_title(task_id),
        family=family,
        requirement=requirement,
        average_steps=average_steps,
        objects=objects,
        generation_success_percent=dict(zip(EMBODIMENT_IDS, rates, strict=True)),
        video_embodiments=video_embodiments,
        official_page=f"https://robotwin-platform.github.io/doc/tasks/{task_id}.html",
        source_code=(
            "https://github.com/RoboTwin-Platform/RoboTwin/blob/"
            f"{ROBOTWIN_REVISION}/envs/{task_id}.py"
        ),
    )


NO_PIPER = ("aloha-agilex", "ARX-X5", "franka-panda", "ur5-wsg")
NO_FRANKA = ("aloha-agilex", "ARX-X5", "piper", "ur5-wsg")

TASKS = (
    _task("adjust_bottle", "reorientation", "Pick up the bottle on the table headup with the correct arm.", 147, ("001_bottle",), (93, 94, 34, 0, 12), NO_PIPER),
    _task("beat_block_hammer", "contact", "There is a hammer and a block on the table; use an arm to grab the hammer and beat the block.", 113, ("020_hammer", "block"), (64, 93, 98, 15, 90)),
    _task("blocks_ranking_rgb", "ordering", "Place the red, green, and blue blocks in that order from left to right, in one row.", 466, ("block",), (96, 97, 99, 13, 53)),
    _task("blocks_ranking_size", "ordering", "Move three randomly colored blocks to the table center and arrange them from largest to smallest, left to right.", 466, ("block",), (96, 97, 89, 7, 38)),
    _task("click_alarmclock", "contact", "Click the center of the alarm clock's top-side button.", 85, ("046_alarm-clock",), (92, 99, 100, 0, 95), NO_PIPER),
    _task("click_bell", "contact", "Click the bell's top center.", 85, ("050_bell",), (100, 100, 100, 91, 100)),
    _task("dump_bin_bigbin", "bimanual", "Grab the small bin and pour the balls into the large bin.", 265, ("011_dustbin", "063_tabletrashbin"), (84, 100, 84, 9, 80)),
    _task("grab_roller", "affordance", "Use both arms to grab the roller on the table.", 94, ("102_roller",), (95, 69, 99, 7, 81)),
    _task("handover_block", "bimanual", "Use the left arm to grasp the red block, hand it to the right arm, and place it on the blue pad.", 283, ("block",), (83, 81, 0, 44, 0), ("aloha-agilex", "ARX-X5", "piper")),
    _task("handover_mic", "bimanual", "Use one arm to grasp the microphone and hand it over to the other arm.", 223, ("018_microphone",), (87, 98, 84, 65, 14)),
    _task("hanging_mug", "affordance", "Pick the mug with the left arm, rotate and place it at the table center, then pick it with the right arm and hang it on the rack.", 340, ("039_mug", "040_rack"), (63, 73, 11, 0, 11), NO_PIPER),
    _task("lift_pot", "bimanual", "Use both arms to lift the pot.", 112, ("060_kitchenpot",), (27, 50, 36, 31, 40)),
    _task("move_can_pot", "placement", "Pick up the can with one arm and move it beside the pot.", 151, ("060_kitchenpot", "105_sauce-can"), (93, 65, 92, 96, 99)),
    _task("move_pillbottle_pad", "placement", "Use one arm to pick up the pill bottle and place it onto the pad.", 147, ("080_pillbottle", "block"), (67, 90, 69, 47, 86)),
    _task("move_playingcard_away", "placement", "Pick up the playing card and move it farther toward the table side where it started.", 120, ("081_playingcards",), (99, 100, 100, 63, 66)),
    _task("move_stapler_pad", "placement", "Use the appropriate arm to move the stapler onto a colored mat.", 152, ("048_stapler",), (92, 96, 89, 13, 75)),
    _task("open_laptop", "articulation", "Use one arm to open the laptop.", 258, ("015_laptop",), (82, 92, 77, 23, 51)),
    _task("open_microwave", "articulation", "Use one arm to open the microwave.", 537, ("044_microwave",), (96, 80, 59, 2, 23)),
    _task("pick_diverse_bottles", "bimanual", "Pick up one bottle with one arm and another bottle with the other arm.", 122, ("001_bottle",), (51, 2, 0, 27, 4), NO_FRANKA),
    _task("pick_dual_bottles", "bimanual", "Pick up one bottle with one arm and another bottle with the other arm.", 127, ("001_bottle",), (92, 6, 0, 81, 7), NO_FRANKA),
    _task("place_a2b_left", "placement", "Use the appropriate arm to place object A to the left of object B.", 155, ("047_mouse", "048_stapler", "050_bell", "057_toycar", "073_rubikscube", "075_bread", "077_phone", "081_playingcards", "086_woodenblock", "107_soap", "112_tea-box", "113_coffee-box"), (80, 88, 64, 28, 76)),
    _task("place_a2b_right", "placement", "Use the appropriate arm to place object A to the right of object B.", 145, ("047_mouse", "048_stapler", "050_bell", "057_toycar", "073_rubikscube", "075_bread", "077_phone", "081_playingcards", "086_woodenblock", "107_soap", "112_tea-box", "113_coffee-box"), (81, 82, 64, 31, 66)),
    _task("place_bread_basket", "placement", "Put one bread in the basket with one arm, or simultaneously put two breads in the basket with both arms when two are present.", 231, ("075_bread", "076_breadbasket"), (89, 88, 62, 1, 67)),
    _task("place_bread_skillet", "placement", "Use one arm to grab the bread and put it into the skillet.", 162, ("075_bread", "106_skillet"), (34, 26, 42, 0, 37), NO_PIPER),
    _task("place_burger_fries", "placement", "Use both arms to pick up the hamburger and fries and put them onto the tray.", 242, ("005_french-fries", "006_hamburg", "008_tray"), (97, 98, 80, 36, 74)),
    _task("place_can_basket", "placement", "Pick up the can and put it in the basket with one arm, then lift the basket with the other arm.", 255, ("071_can", "110_basket"), (70, 28, 61, 0, 3), NO_PIPER),
    _task("place_cans_plasticbox", "placement", "Use both arms to pick up cans and place them into the plastic box.", 289, ("062_plasticbox", "071_can"), (100, 96, 85, 0, 82), NO_PIPER),
    _task("place_container_plate", "placement", "Place the container onto the plate.", 156, ("002_bowl", "003_plate", "021_cup"), (89, 86, 86, 37, 81)),
    _task("place_dual_shoes", "bimanual", "Use both arms to put the two shoes into the shoe box with their toes pointing left.", 228, ("007_shoe-box", "041_shoe"), (77, 31, 41, 1, 32)),
    _task("place_empty_cup", "placement", "Use one arm to place the empty cup on the coaster.", 174, ("019_coaster", "021_cup"), (92, 100, 100, 4, 100)),
    _task("place_fan", "placement", "Place the fan on a colored mat and make sure it faces the robot.", 148, ("099_fan", "block"), (95, 93, 83, 0, 65), NO_PIPER),
    _task("place_mouse_pad", "placement", "Grab the mouse and place it on a colored mat.", 149, ("047_mouse", "block"), (99, 89, 100, 23, 73)),
    _task("place_object_basket", "placement", "Put the target object in the basket, grasp the basket with the other arm, and move the basket slightly away.", 252, ("057_toycar", "081_playingcards", "110_basket"), (74, 14, 61, 0, 7), NO_PIPER),
    _task("place_object_scale", "placement", "Use one arm to grab the object and put it on the scale.", 146, ("047_mouse", "048_stapler", "050_bell", "072_electronicscale"), (78, 92, 82, 2, 76)),
    _task("place_object_stand", "placement", "Use the appropriate arm to place the object on the stand.", 138, ("047_mouse", "048_stapler", "050_bell", "057_toycar", "073_rubikscube", "074_displaystand", "079_remotecontrol"), (97, 99, 81, 9, 92)),
    _task("place_phone_stand", "placement", "Pick up the phone and put it on the phone stand.", 130, ("077_phone", "078_phonestand"), (66, 78, 45, 53, 49)),
    _task("place_shoe", "placement", "Use one arm to grab the shoe and place it on the mat.", 178, ("041_shoe", "block"), (84, 85, 74, 7, 91)),
    _task("press_stapler", "contact", "Use one arm to press the stapler.", 141, ("048_stapler",), (98, 96, 100, 59, 72)),
    _task("put_bottles_dustbin", "placement", "Use both arms to put the bottles into the dustbin on the left side of the table.", 637, ("011_dustbin", "114_bottle"), (71, 1, 0, 56, 0), ("aloha-agilex", "ARX-X5", "piper")),
    _task("put_object_cabinet", "placement", "Open the cabinet drawer with one arm and put the table object into it with the other arm.", 274, ("036_cabinet", "047_mouse", "048_stapler", "057_toycar", "073_rubikscube", "075_bread", "077_phone", "081_playingcards", "107_soap", "112_tea-box", "113_coffee-box"), (14, 24, 55, 0, 0), ("aloha-agilex", "ARX-X5", "franka-panda")),
    _task("rotate_qrcode", "reorientation", "Pick up the QR-code board and rotate it until the QR code faces the robot.", 155, ("070_paymentsign",), (75, 74, 94, 0, 67), NO_PIPER),
    _task("scan_object", "contact", "Pick up the scanner with one arm, the object with the other, and scan the object.", 170, ("024_scanner", "112_tea-box"), (4, 45, 26, 0, 19), NO_PIPER),
    _task("shake_bottle", "reorientation", "Shake the bottle with the appropriate arm.", 246, ("001_bottle",), (89, 94, 85, 74, 97)),
    _task("shake_bottle_horizontally", "reorientation", "Shake the bottle horizontally with the appropriate arm.", 276, ("001_bottle",), (90, 94, 85, 74, 98)),
    _task("stack_blocks_three", "ordering", "Move three blocks to the table center and stack blue on green, then green on red.", 481, ("block",), (94, 96, 80, 0, 51), NO_PIPER),
    _task("stack_blocks_two", "ordering", "Move the red and green blocks to the table center and stack green on red.", 316, ("block",), (98, 99, 96, 2, 68)),
    _task("stack_bowls_three", "ordering", "Stack the three bowls on top of each other.", 476, ("002_bowl",), (43, 57, 82, 0, 81), NO_PIPER),
    _task("stack_bowls_two", "ordering", "Stack the two bowls on top of each other.", 313, ("002_bowl",), (78, 82, 88, 4, 94)),
    _task("stamp_seal", "contact", "Grab the stamp and stamp it onto the specified colored mat.", 151, ("100_seal", "block"), (56, 91, 4, 37, 100)),
    _task("turn_switch", "articulation", "Use one arm to click the switch.", 95, ("056_switch",), (74, 3, 36, 81, 10)),
)

TASK_BY_ID = {task.id: task for task in TASKS}
if len(TASKS) != 50 or len(TASK_BY_ID) != 50:
    raise RuntimeError("RoboTwin gallery snapshot must contain 50 unique tasks")


def catalog_payload() -> dict:
    """Return the stable read-only JSON representation served by the viewer."""

    return {
        "schema": CATALOG_SCHEMA,
        "robotwin_revision": ROBOTWIN_REVISION,
        "robotwin_doc_revision": ROBOTWIN_DOC_REVISION,
        "migrated_from": {
            "repository": "ora-agent",
            "commit": MIGRATED_FROM_COMMIT,
        },
        "media": {
            "kind": "official_scripted-expert-example",
            "root": ROBOTWIN_MEDIA_ROOT,
            "viewpoints": ["world", "head"],
            "resolution": [320, 240],
        },
        "embodiments": [
            {"id": identifier, "label": label} for identifier, label in EMBODIMENTS
        ],
        "families": [
            {"id": identifier, "label_zh": labels[0], "label_en": labels[1]}
            for identifier, labels in FAMILY_LABELS.items()
        ],
        "tasks": [asdict(task) for task in TASKS],
    }
