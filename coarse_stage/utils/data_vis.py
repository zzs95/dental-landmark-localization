import matplotlib.pyplot as plt

def plot_img_and_mask(img, mask):
    plt.figure()
    # a = fig.add_subplot(1, 2, 1)
    # a.set_title('Input image')
    plt.imshow(img)

    # b = fig.add_subplot(1, 2, 2)
    # b.set_title('Output mask')
    for i in range(mask.shape[0]):
        plt.figure()
        plt.imshow(mask[i], plt.cm.gray)
    plt.show()