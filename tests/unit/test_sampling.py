from perceptiongate.data.sampling import allocate, stratified_sample
from perceptiongate.data.schema import Box, Frame

BOX = (Box("car", 0, 0, 10, 10),)


def frames(counts: dict[str, int]) -> list[Frame]:
    out = []
    for weather, n in counts.items():
        out += [Frame(name=f"{weather}_{i}.jpg", weather=weather, boxes=BOX) for i in range(n)]
    return out


def test_allocate_sums_to_n_and_respects_capacity() -> None:
    sizes = {("clear",): 900, ("rainy",): 80, ("foggy",): 5}
    alloc = allocate(sizes, 200, floor=20)
    assert sum(alloc.values()) == 200
    assert alloc[("foggy",)] == 5  # all it has
    assert alloc[("rainy",)] >= 20


def test_rare_strata_get_their_floor() -> None:
    sample = stratified_sample(
        frames({"clear": 1000, "foggy": 30}), n=100, keys=["weather"], min_per_stratum=25, seed=1
    )
    assert len(sample) == 100
    assert sum(f.weather == "foggy" for f in sample) >= 25


def test_sampling_is_deterministic() -> None:
    f = frames({"clear": 500, "rainy": 100})
    a = stratified_sample(f, 50, ["weather"], 5, seed=7)
    b = stratified_sample(list(reversed(f)), 50, ["weather"], 5, seed=7)
    assert [x.name for x in a] == [x.name for x in b]


def test_frames_without_boxes_are_skipped() -> None:
    f = [Frame(name="empty.jpg"), Frame(name="full.jpg", boxes=BOX)]
    assert [x.name for x in stratified_sample(f, 2, ["weather"], 0, seed=0)] == ["full.jpg"]


def test_asking_for_more_than_exists_returns_everything() -> None:
    f = frames({"clear": 3, "rainy": 2})
    assert len(stratified_sample(f, 100, ["weather"], 1, seed=0)) == 5
