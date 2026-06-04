"""
Project code+scripts for 8BE030 course
"""

import numpy as np
import segmentation_util as util
import matplotlib.pyplot as plt
import segmentation as seg
import scipy

def segmentation_mymethod(train_data_matrix, train_labels_matrix, test_data, task='brain'):
    all_predictions = []

    num_subjects = train_labels_matrix.shape[1]

    for i in range(num_subjects):
        train_data = train_data_matrix[:, :, i]
        train_labels = train_labels_matrix[:, i].astype(int)

        pred_k1 = seg.segmentation_knn(train_data, train_labels, test_data, k=1)
        pred_k3 = seg.segmentation_knn(train_data, train_labels, test_data, k=3)
        pred_k5 = seg.segmentation_knn(train_data, train_labels, test_data, k=5)

        all_predictions.append(pred_k1.astype(int))
        all_predictions.append(pred_k3.astype(int))
        all_predictions.append(pred_k5.astype(int))

    all_predictions = np.array(all_predictions).T.astype(int)

    predicted_labels = np.zeros(all_predictions.shape[0], dtype=int)

    for j in range(all_predictions.shape[0]):
        values, counts = np.unique(all_predictions[j, :], return_counts=True)
        predicted_labels[j] = values[np.argmax(counts)]

    if task == 'brain':
        predicted_labels = predicted_labels.astype(bool)

    return predicted_labels


def segmentation_demo():

    train_subject = 1
    test_subject = 2
    train_slice = 1
    test_slice = 1
    task = 'brain'

    #Load data
    train_data, train_labels, train_feature_labels = util.create_dataset(train_subject,train_slice,task)
    test_data, test_labels, test_feature_labels = util.create_dataset(test_subject,test_slice,task)

    # predicted_labels = seg.segmentation_knn(None, train_labels, None)

    # err = util.classification_error(test_labels, predicted_labels)
    # dice = util.dice_overlap(test_labels, predicted_labels)

    #Display results
    # true_mask = test_labels.reshape(240, 240)
    # predicted_mask = predicted_labels.reshape(240, 240)

    # fig = plt.figure(figsize=(8,8))
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(true_mask, 'gray')
    # ax1.imshow(predicted_mask, 'viridis', alpha=0.5)
    # print('Subject {}, slice {}.\nErr {}, dice {}'.format(test_subject, test_slice, err, dice))

    ## Compare methods
    num_images = 5
    num_methods = 3
    im_size = [240, 240]

    all_errors = np.empty([num_images,num_methods])
    all_errors[:] = np.nan
    all_dice = np.empty([num_images,num_methods])
    all_dice[:] = np.nan

    all_subjects = np.arange(num_images)
    train_slice = 1
    task = 'brain'
    all_data_matrix = np.empty([train_data.shape[0],train_data.shape[1],num_images])
    all_labels_matrix = np.empty([train_labels.size,num_images], dtype=bool)

    #Load datasets once
    print('Loading data for ' + str(num_images) + ' subjects...')

    for i in all_subjects:
        sub = i+1
        train_data, train_labels, train_feature_labels = util.create_dataset(sub,train_slice,task)
        all_data_matrix[:,:,i] = train_data
        all_labels_matrix[:,i] = train_labels.flatten()

    print('Finished loading data.\nStarting segmentation...')

    #Go through each subject, taking i-th subject as the test
    for i in np.arange(num_images):
        sub = i+1
        #Define training subjects as all, except the test subject
        train_subjects = all_subjects.copy()
        train_subjects = np.delete(train_subjects, i)

        train_data_matrix = all_data_matrix[:,:,train_subjects]
        train_labels_matrix = all_labels_matrix[:,train_subjects]
        test_data = all_data_matrix[:,:,i]
        test_labels = all_labels_matrix[:,i]
        test_shape_1 = test_labels.reshape(im_size[0],im_size[1])

        fig = plt.figure(figsize=(10,5))

        predicted_labels = seg.segmentation_combined_knn(train_data_matrix,train_labels_matrix,test_data)
        all_errors[i,0] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,0] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_1 = predicted_labels.reshape(im_size[0],im_size[1])
        ax1 = fig.add_subplot(121)
        ax1.imshow(test_shape_1, 'gray')
        ax1.imshow(predicted_mask_1, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,0], all_dice[i,0])
        ax1.set_xlabel(text_str)
        ax1.set_title('Subject {}: Combined k-NN'.format(sub))

        predicted_labels = segmentation_mymethod(train_data_matrix,train_labels_matrix,test_data,task)
        all_errors[i,1] = util.classification_error(test_labels, predicted_labels)
        all_dice[i,1] = util.dice_overlap(test_labels, predicted_labels)
        predicted_mask_2 = predicted_labels.reshape(im_size[0],im_size[1])
        ax2 = fig.add_subplot(122)
        ax2.imshow(test_shape_1, 'gray')
        ax2.imshow(predicted_mask_2, 'viridis', alpha=0.5)
        text_str = 'Err {:.4f}, dice {:.4f}'.format(all_errors[i,1], all_dice[i,1])
        ax2.set_xlabel(text_str)
        ax2.set_title('Subject {}: My method'.format(sub))



def add_salt_pepper_noise(labels, noise_fraction=0.05, n_classes=4):
    """
    Verandert willekeurig een percentage van de labels.

    Hiermee simuleren we fouten in de segmentatie zodat
    het effect van MRF beter zichtbaar wordt.
    """

    noisy_labels = labels.copy()

    n_pixels = len(labels)
    n_noisy = int(noise_fraction * n_pixels)

    idx = np.random.choice(n_pixels, n_noisy, replace=False)

    noisy_labels[idx] = np.random.randint(
        0,
        n_classes,
        size=n_noisy
    )

    return noisy_labels


def mrf_regularization(
        labels,
        image_shape=(240,240),
        n_classes=4,
        beta=1.0, #sterkte van de smoothness term 0 = geen regularisatie, >0 = meer regularisatie
        n_iter=5): #aantal iteraties van het algoritme, meer iteraties = meer kans op convergentie, maar ook meer rekentijd
                    #verandering van pixel A heeft invloed op pixel B daarom ook meerdere iteraties nodig
    original = labels.reshape(image_shape)

    current = original.copy()

    rows, cols = image_shape

    for iteration in range(n_iter):

        for r in range(rows): #ga door alle pixels heen

            for c in range(cols): #ga door alle pixels heen

                best_label = current[r,c] #start met huidige label als beste label, dit is nodig voor het geval dat alle andere labels een hogere energie hebben
                best_energy = np.inf

                for candidate in range(n_classes): #ga door alle mogelijke labels heen, en bereken de energie van elk label, kies het label met de laagste energie als beste label

                    # DATA TERM

                    data_energy = (
                        candidate != original[r,c] # de data term is 0 als het candidate label gelijk is aan het originele label, en 1 als het candidate label verschillend is van het originele label
                    )

                    # SMOOTHNESS TERM

                    smooth_energy = 0

                    if r > 0:
                        smooth_energy += (
                            candidate != current[r-1,c]
                        ) #bovenste buurman

                    if r < rows-1:
                        smooth_energy += (
                            candidate != current[r+1,c]
                        ) #onderste buurman

                    if c > 0:
                        smooth_energy += (
                            candidate != current[r,c-1]
                        ) #linker buurman

                    if c < cols-1:
                        smooth_energy += (
                            candidate != current[r,c+1]
                        ) #rechter buurman

                        #we kunnen eventueel ook nog diagonaal buurman meenemen even kijken wat dat doet met de randen. 

                    energy = (
                        data_energy +
                        beta*smooth_energy
                    )

                    if energy < best_energy:
                        best_energy = energy
                        best_label = candidate

                current[r,c] = best_label

    return current.flatten()

