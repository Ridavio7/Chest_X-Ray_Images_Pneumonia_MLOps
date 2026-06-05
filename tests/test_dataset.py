from src.data.dataset import XRayDataset


def test_dataset_not_empty():

    dataset = XRayDataset(
        "data/chest_xray/train"
    )

    assert len(dataset) > 0