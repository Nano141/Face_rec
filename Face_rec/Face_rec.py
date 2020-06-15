import os
import cv2
import face_recognition

print('creating known people dataset....') # working on known images
Faces = []
Names = []
for name in os.listdir('known_faces'): # we put all known pictures in a folder named with the persons name
    # folder name must be correct as this is what will show up whenever that person is detected
    
    for img in os.listdir(f'known_faces/{name}'): 
        # I loop on all folders in the known directory using f string path

        # Load an image
        image = face_recognition.load_image_file(f'known_faces/{name}/{img}') 
        #I loop on all pictures in a persons folder directory using f string path

        
        #Here I get the 128 bit encoding of the main face in the picture ([0] is used in case a picture with people in the background was used)
        #[0] prevents the encoding of more than one face
        enc = face_recognition.face_encodings(image)[0]

        Faces.append(enc)
        Names.append(name)


print('Working on it :)....') # working on unknown images
for img in os.listdir('unknown_faces'):

    # Looping over images in unknown folder
    image = face_recognition.load_image_file(f'unknown_faces/{img}')

    # here I find the location of the detected face
    location = face_recognition.face_locations(image)

    # I get the encodings of the faces in the images using the face location to make things easier and faster
    encoding = face_recognition.face_encodings(image, location)

    # I added this line here to make cv2 more efficient
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # In case we have more than one face in the image
    for face_encoding, face_location in zip(encoding, location):

        # Compare face encoding against another encoding to know who it is
        # it classifies with 99.38% accuracy using dlib library
        results = face_recognition.compare_faces(Faces, face_encoding, 0.6)
        #results = face_recognition.face_distance(knownFaces,face_encoding,0.6)

        match = None
        if True in results:  # if image encoding is equivalent to at least one known picture
            match = Names[results.index(True)]
            print("Detected %s's face"%Names[results.index(True)])

            # I use cv2 to draw a rectanle around the detected face and write the name under the rectangle
            # Positions in order: top[0], right[1], bottom[2], left[3]
            top_left = (face_location[3]-10, face_location[0]-10)
            bottom_right = (face_location[1]+10, face_location[2]+10)

            color = (0, 0, 139)

            cv2.rectangle(image, top_left, bottom_right, color, 1)

            top_left = (face_location[3]-10, face_location[2]+10)
            bottom_right = (face_location[1]+10, face_location[2] + 30)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3]-10, face_location[2] + 25), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)


    cv2.imshow(img, image)
    cv2.waitKey(0)
    cv2.destroyWindow(img)
