from datasets import *
from vit import *
from engine import *

def main():
    #seed_everything
    seed_everything(43)
    
    # load dataset
    train_df = data_preprocessing(TRAIN_PATH)
    valid_df = data_preprocessing(VALID_PATH)

    train_dataset = PneumoniaDataset(train_df, transforms_train)
    valid_dataset = PneumoniaDataset(valid_df, transforms_val)


    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )


    # load model and loss
    vit_base_model = ViTBase16(n_classes=1, pretrained=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(vit_base_model.parameters(), lr=LR)

    # fit
    print('============================== TRAINING ==============================')
    history = fit(vit_base_model, train_loader, valid_loader, N_EPOCHS, criterion, optimizer, DEVICE)
    print('======================================================================')
    # saving history
    with open(HISTORY_PATH, 'wb') as file:
        pkl.dump(history, file)

if __name__ == '__main__':
    main()