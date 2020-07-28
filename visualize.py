# model.summary()
def load_image_ben_orig(image,resize=True,crop=False,norm255=True,keras=False):
    #image = cv2.imread(path)
    
#     if crop:
#         image = crop_image(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#     if resize:
#         image = cv2.resize(image,(SIZE,SIZE))
        
    image=cv2.addWeighted( image,4, cv2.GaussianBlur( image , (0,0) ,  10) ,-4 ,128)
#     image=cv2.addWeighted( image,4, cv2.medianBlur( image , 10) ,-4 ,128)
    
    # NOTE plt.imshow can accept both int (0-255) or float (0-1), but deep net requires (0-1)
    if norm255:
        return image/255
    elif keras:
        #see https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py for mode
        #see https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py for inception,xception mode
        #the use of tf based preprocessing (- and / by 127 respectively) will results in [-1,1] so it will not visualize correctly (directly)
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)[0]
    else:
        return image.astype(np.int16)
    
    return image

def transform_image_ben(img,resize=True,crop=False,norm255=True,keras=False):  
    image=cv2.addWeighted( img,4, cv2.GaussianBlur( img , (0,0) ,  10) ,-4 ,128)
    
    # NOTE plt.imshow can accept both int (0-255) or float (0-1), but deep net requires (0-1)
    if norm255:
        return image/255
    elif keras:
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)[0]
    else:
        return image.astype(np.int16)
    
    return image

def gen_heatmap_img(img, model0, layer_name='last_conv_layer',viz_img=None,orig_img=None):
    preds_raw = model0.predict(img[np.newaxis])
    preds = preds_raw > 0.5 # use the same threshold as @xhlulu original kernel
    class_idx = (preds.astype(int).sum(axis=1) - 1)[0]
#     print(class_idx, class_idx.shape)
    class_output_tensor = model0.output[:, class_idx]
    
    viz_layer = model0.get_layer(layer_name)
    grads = K.gradients(
                        class_output_tensor ,
                        viz_layer.output
                        )[0] # gradients of viz_layer wrt output_tensor of predicted class
    
    pooled_grads=K.mean(grads,axis=(0,1,2))
    iterate=K.function([model0.input],[pooled_grads, viz_layer.output[0]])
    
    pooled_grad_value, viz_layer_out_value = iterate([img[np.newaxis]])
    
    for i in range(pooled_grad_value.shape[0]):
        viz_layer_out_value[:,:,i] *= pooled_grad_value[i]
    
    heatmap = np.mean(viz_layer_out_value, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)

    viz_img=cv2.resize(viz_img,(img.shape[1],img.shape[0]))
    heatmap=cv2.resize(heatmap,(viz_img.shape[1],viz_img.shape[0]))
    
    heatmap_color = cv2.applyColorMap(np.uint8(heatmap*255), cv2.COLORMAP_SPRING)/255
    heated_img = heatmap_color*0.5 + viz_img*0.5
    
    print('raw output from model : ')
    print_pred(preds_raw)
    
    if orig_img is None:
        show_Nimages([img,viz_img,heatmap_color,heated_img])
    else:
        show_Nimages([orig_img,img,viz_img,heatmap_color,heated_img])
    
    plt.show()
    return heated_img

def visualize(img):
    ben_img = load_image_ben_orig(img)
    input_img = np.empty((1,224, 224, 3), dtype=np.uint8)
    input_img[0,:,:,:] = preprocess_image(path)
    heated_img = gen_heatmap_img(input_img[0],
                        model, layer_name=layer_name,viz_img=ben_img)

    return ben_img,input_img,heated_img