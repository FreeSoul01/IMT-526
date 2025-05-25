from fairface_eval import predict_image

def gender_balancing_filter(gender_counts, images, model):
    male_ratio = gender_counts['Male'] / (gender_counts['Male'] + gender_counts['Female'])
    threshold = 0.5
    keep_images = []
    for img_path in images:
        gender, _ = predict_image(model, img_path)
        if male_ratio > threshold and gender == 'Male':
            continue
        elif male_ratio < threshold and gender == 'Female':
            continue
        keep_images.append(img_path)
    return keep_images