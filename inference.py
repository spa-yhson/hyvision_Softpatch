import argparse
import os
import pickle

def load_pickle(file_path):
    """Load and return the content of the pickle file."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Load and process a pickle file.")
    parser.add_argument('--file-path', type=str, help='Path to the pickle file', required=True)

    # Parse arguments
    args = parser.parse_args()

    # Check if file exists
    if not os.path.exists(args.file_path):
        print(f"The file {args.file_path} does not exist.")
        return

    # Load the pickle file
    data = load_pickle(args.file_path)

    num_cls = 15
    mean_img_auroc = 0
    mean_pix_auroc = 0

    img_auroc_val_list = []
    pix_auroc_val_list = []

    for idx, key_name in enumerate(data.keys()):
        tmp_key = key_name[-9:]

        if tmp_key == 'img_auroc':
            mean_img_auroc += data[key_name]
            img_auroc_val_list.append(round(data[key_name], 4))
        else:
            mean_pix_auroc += data[key_name]
            pix_auroc_val_list.append(round(data[key_name], 4))

    print("IMG_AUTOC")
    for val in img_auroc_val_list:
        print(val)

    print("PIXEL_AUTOC")
    for val in pix_auroc_val_list:
        print(val)
    

    # Final Mean AUROC
    print("mean_img_auroc: ", round(mean_img_auroc/num_cls, 4))
    print("mean_pix_auroc: ", round(mean_pix_auroc/num_cls, 4))


if __name__ == "__main__":
    main()
