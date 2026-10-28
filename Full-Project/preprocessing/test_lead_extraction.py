from skimage.io import imread
from skimage import color
from skimage.transform import resize
import matplotlib.pyplot as plt


# -----------------------------
# 1. Load ECG image
# -----------------------------
image_path = "dataset/MI/MI(1).jpg"

image = imread(image_path)

print("Original image shape:", image.shape)


# -----------------------------
# 2. Convert to grayscale
# -----------------------------
gray_image = color.rgb2gray(image)

# Reference pipeline uses 1572 x 2213
gray_image = resize(gray_image, (1572, 2213))

print("Grayscale shape:", gray_image.shape)


# -----------------------------
# 3. Divide into ECG leads
# -----------------------------
leads = {}

leads["Lead_1"] = gray_image[300:600, 150:643]
leads["Lead_2"] = gray_image[300:600, 646:1135]
leads["Lead_3"] = gray_image[300:600, 1140:1625]
leads["Lead_4"] = gray_image[300:600, 1630:2125]

leads["Lead_5"] = gray_image[600:900, 150:643]
leads["Lead_6"] = gray_image[600:900, 646:1135]
leads["Lead_7"] = gray_image[600:900, 1140:1625]
leads["Lead_8"] = gray_image[600:900, 1630:2125]

leads["Lead_9"] = gray_image[900:1200, 150:643]
leads["Lead_10"] = gray_image[900:1200, 646:1135]
leads["Lead_11"] = gray_image[900:1200, 1140:1625]
leads["Lead_12"] = gray_image[900:1200, 1630:2125]

# Long lead
leads["Lead_13"] = gray_image[1250:1480, 150:2125]


# -----------------------------
# 4. Print dimensions
# -----------------------------
for name, lead in leads.items():
    print(name, ":", lead.shape)


# -----------------------------
# 5. Display the 13 leads
# -----------------------------
fig, axes = plt.subplots(13, 1, figsize=(12, 20))

for ax, (name, lead) in zip(axes, leads.items()):
    ax.imshow(lead, cmap="gray")
    ax.set_title(name)
    ax.axis("off")

plt.tight_layout()
plt.show()