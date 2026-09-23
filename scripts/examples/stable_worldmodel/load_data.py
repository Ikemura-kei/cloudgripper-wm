import stable_worldmodel as swm

def main():
    # load data from disk
    dataset = swm.data.load_dataset(
        name='tutorial_pusht.lance',
        num_steps=4,
        frameskip=1,
        keys_to_load=['pixels', 'action', 'state']
    )

    print(len(dataset))
    # sample : dict = dataset[0]
    # print(sample.keys())
    # print(sample['pixels'])
    # print(sample['action'])
    # print(sample['state'])


    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    batch = next(iter(loader))
    print(batch['pixels'].shape)


if __name__ == '__main__':
    main()