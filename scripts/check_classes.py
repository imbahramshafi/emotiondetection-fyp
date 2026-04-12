from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = "datasets/fer2013/train"

gen = ImageDataGenerator(rescale=1/255.0)
flow = gen.flow_from_directory(train_dir, target_size=(48,48), color_mode="grayscale")

print("Class index to label mapping:")
print(flow.class_indices)
