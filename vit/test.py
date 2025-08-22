from configuration import BEST_MODEL_PATH
from datasets import *
from vit import *
from engine import *

def main():
    # load dataset
    test_df = data_preprocessing(TEST_PATH)
    test_dataset = PneumoniaDataset(test_df, transforms_val)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKS
    )

    # load best model
    best_model = ViTBase16(n_classes=1, pretrained=True).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=LR)

    print('============================== TESTING ==============================')
    test_loss, test_acc, test_pre, test_rec = validate_one_epoch(best_model, test_loader, criterion, DEVICE)
    print(f"Test loss: {test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_pre}, Test Recall: {test_rec}")
    print('=====================================================================')

if __name__ == '__main__':
    main()