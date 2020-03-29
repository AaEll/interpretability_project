from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from keras import backend as K
import keras
import numpy as np
import boto3
import io
import pickle

#Load the VGG model    
model = vgg16.VGG16(weights='imagenet')

s3_client = boto3.client('s3')

bucket_arn = 'arn:aws:s3:::ai-output-dump'

layer_names = [layer.name for layer in model.layers]

test_image_path = 'data/flickr30k_images/flickr30k_images/flickr30k_images/996089206.jpg'

def display_image(image_path):
    import cv2
    
    image = cv2.imread(image_path)
    img = cv2.imread(test_image_path)
    img2 = img[:,:,::-1]
    plt.imshow(img2)

#----- flow for single image
def load_image(image_path):
    image = image_utils.load_img(image_path, target_size=(224, 224))
    image = image_utils.img_to_array(image)

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def get_intermediate_model(layer_name, model):
    intermediate_layer_model = Model(inputs=model.input,
                             outputs=model.get_layer(layer_name).output)
    return intermediate_layer_model
    
def predict_image(model,image):
    preds = model.predict(image)
    #preds = decode_predictions(preds)
    return preds
# trunc_model = get_intermediate_model(layer_name, full_model)
# img = load_image(image_path)
# out = predict_image(trunc_model,img)
#-----

def get_image_flow(data_path, batch_size = 32, target_size = (224,224), **kwargs):
    datagen = ImageDataGenerator()
    flow = datagen.flow_from_directory(data_path,
                                       target_size=target_size,
                                       batch_size=batch_size,
                                       **kwargs)
    return flow

def predict_flow(model,flow,steps = 1, **kwargs):
    return model.predict_generator(generator=flow,steps = steps,**kwargs)

# trunc_model = get_intermediate_model(layer_name, full_model)
# flow = get_image_flow('data/flickr30k_images/flickr30k_images/')
# out = predict_flow(trunc_model, flow)

def get_distance_matrix(model_outputs):
    from sklearn.metrics.pairwise import euclidean_distances
    return euclidean_distances(model_outputs,model_outputs)

def draw_heatmap(distance_matrix, ax = None):
    import seaborn as sns
    ax = sns.heatmap(distance_matrix,cmap="inferno", ax = ax)
    return ax

# distance_matrix = get_distance_matrix(out)
# draw_heatmap(distance_matrix)


def write_to_s3(np_array, s3_bucket, filename):
    array_data = io.BytesIO()
    pickle.dump(np_array, array_data)
    array_data.seek(0)
    s3_client.upload_fileobj(array_data, s3_bucket, filename)



#----


for layer_name in layer_names[20:]:
    trunc_model = get_intermediate_model(layer_name, model)
    flow = get_image_flow('data/flickr30k_images/flickr30k_images/')
    out = predict_flow(trunc_model, flow, steps = 1000)
    distance_matrix = get_distance_matrix(out)
    write_to_s3(out,'ai-output-dump', layer_name+'_output_45.pkl')
    
