# Import libraries -----------------------------
import cv2
import datetime
import numpy as np
import time
import os


def locate_video_file(current_date, path):
    video_file_path = path
    count = 0
    for filename in os.listdir(video_file_path):
        if current_date in filename:
            count += 1

    return str(count+1)


def run_program():

    # Get a video capture object
    video = cv2.VideoCapture(0)    

    # Frame width
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Frame height
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Capture video path
    path = "C:/Users/giorg/aiVenv/Security_Camera/Video_Captures/"


    # Load the pre-trained model
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Detection state
    writer_initiated = False

    writer = None

    # Timer state
    start_timer_initiated = False

    # Define video codec
    fourcc = cv2.VideoWriter_fourcc(*'XVID')



    while(True):

        # Capture the video frame by frame
        ret, frame = video.read()

        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect face 
        face = face_classifier.detectMultiScale(gray_frame, 1.1, 4)

        # If a face is detected -> True
        # If a face is not detected -> False
        detected_face = np.any(face)

        if detected_face:
        
            # If the writer object has not been created
            if not writer_initiated:
                # Get the current time and date
                current_date = str(datetime.datetime.now().date().strftime("%d_%m_%Y"))

                # Check if video file exists
                video_index = locate_video_file(current_date, path)
                # Create VideoWriter object
                writer = cv2.VideoWriter(path+current_date+"_"+video_index+".avi",fourcc, 20.0, (frame_width,frame_height))

                writer_initiated = True

                print("Started Recording.")

            if start_timer_initiated:
                start_timer_initiated = False

            

        else:
            
            # If timer has not yet started and the writer object has been created
            if not start_timer_initiated and writer_initiated:
                start_time = time.time()
                start_timer_initiated = True
            
            # If the writer object has been created and the elapsed time since the timer started is 5 sec
            if writer_initiated and ((time.time() - start_time) > 5.0) and start_timer_initiated:

                # Release the writer object
                writer.release()

                writer_initiated = False

                start_timer_initiated = False

                print("Stoped Recording.")

        
        # If the writer object has been created
        if writer_initiated:

            # Show time in video
            cv2.putText(frame, str(datetime.datetime.now().time()), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

            writer.write(frame)


        

        # Display each frame
        cv2.imshow("Footage", frame)

        # Break the loop when 'q' button is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the object
    video.release()

    # After the loop release the writer object
    writer.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


# Main program -----------------------------------
if __name__ == "__main__":
    run_program()