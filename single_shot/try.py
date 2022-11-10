import turicreate as tc

# Load the starter images
starter_images = tc.SFrame({'image':[tc.Image('/home/zekun/mineral_competition/data/validation_point/USGS_I-1803-D_1_label_BIF_pt.jpeg')],
                   'label':['stop_sign']})

# Load test images
test_images = tc.SFrame({'image':[tc.Image('/data2/mineral_competition/data/validation/USGS_I-1803-D_1.tif'), ]})

# Create a model. This step will take a few hours on CPU and about an hour on GPU
model = tc.one_shot_object_detector.create(starter_images, 'label')

# Save predictions on the test set
test_images['predictions'] = model.predict(test_images)

# Draw prediction bounding boxes on the test images
test_images['annotated_predictions'] = \
    tc.one_shot_object_detector.util.draw_bounding_boxes(test_images['image'],
        test_images['predictions']) 

# To visualize the predictions made on the test set
test_images.explore()

# Save the model for later use in TuriCreate
model.save('stop-sign.model')

# Export for use in Core ML
model.export_coreml('MyCustomOneShotDetector.mlmodel')