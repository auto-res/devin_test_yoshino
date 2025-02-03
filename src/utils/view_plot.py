import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def view_plot(plot_path):
    img = mpimg.imread(plot_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python view_plot.py <plot_path>")
        sys.exit(1)
    view_plot(sys.argv[1])
