import pandas as pd
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from clarifai_grpc.grpc.api import resources_pb2, service_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from google.protobuf import json_format
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
from google.protobuf.json_format import MessageToDict
import requests
import cv2, urllib, PIL
from PIL import Image
from io import BytesIO
import requests
import streamlit as st
import pandas as pd
import pandas as pd

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
userDataClarifaiMain= resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')

def convert_to_seconds(time_str):
    """
    Convert a time string in 'mm:ss' format to seconds.
    """
    if ':' not in time_str:
        return 0  # return 0 if the format is incorrect

    minutes, seconds = time_str.split(':')
    return int(minutes) * 60 + int(seconds)

def display_frame(video_url, frame_index):
    response = requests.get(video_url)
    video_bytes = BytesIO(response.content)
    cap = cv2.VideoCapture(video_bytes)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = cap.read()
    if not success:
        st.error(f"Failed to extract frame at index {frame_index} from the video.")
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f'Frame at index {frame_index}')
    cap.release()


def crop_workflow_predict(file_bytes,
                     WORKFLOW_ID="Face-Detect-Crop"):
    # Crop the face with this workflow to feed to visual search
   
    post_workflow_results_response = stub.PostWorkflowResults(
        service_pb2.PostWorkflowResultsRequest(
            user_app_id=userDataObject,
            workflow_id=WORKFLOW_ID,
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=file_bytes
                        )
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_workflow_results_response

def make_visual_search_req(file_bytes):
    
    post_annotations_searches_response = stub.PostAnnotationsSearches(
        service_pb2.PostAnnotationsSearchesRequest(
            user_app_id=userDataObject,
            searches=[
                resources_pb2.Search(
                    query=resources_pb2.Query(
                        ranks=[
                            resources_pb2.Rank(
                                annotation=resources_pb2.Annotation(
                                    data=resources_pb2.Data(
                                        image=resources_pb2.Image(
                                            base64=file_bytes
                                        )
                                    )
                                )
                            )
                        ]
                    )
                )
            ]
        ),
        metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return post_annotations_searches_response

if __name__ == "__main__":
  st.set_page_config(layout="wide")
  ClarifaiStreamlitCSS.insert_default_css(st)

  # Add an upload widget so that the user can upload an image.
  uploaded_file = st.file_uploader("Upload an image:")
  # Add a slider next to the uploaded_file that allows the user to choose the confidence threshold. Set it between 0.8 and 1.0.
  with st.sidebar:
    confidence_threshold = st.slider("Confidence threshold", 0.6, 1.0, 0.84, 0.01)
  
  if uploaded_file is not None:
    file_bytes = uploaded_file.read()

    # Display the image.
    image = Image.open(io.BytesIO(file_bytes))
    resized_image = image.resize((300, 300))  # Resize the image to a smaller size
    st.image(resized_image, caption='Uploaded Image')

    # Print image dimensions
    image_width, image_height = image.size
    st.write(f"**Image dimensions**: {image_width} x {image_height}")

  # Now predict with the workflow where we detect + crop faces
    detect_faces_crop = crop_workflow_predict(file_bytes)
    if detect_faces_crop.status.code != status_code_pb2.SUCCESS:
        print(detect_faces_crop.status)
        print("Post workflow results failed, status: " + detect_faces_crop.status.description)

    # st.write(detect_faces_crop.results[0].outputs[-1].data.regions)
    detected_faces = detect_faces_crop.results[0].outputs[-1].data.regions
    for face in detected_faces:            
      # SAVE THE CROPPED FACE AS A BASE64 STRING
      st.write('-'*100)
      base_64_img = face.data.image.base64
      st.image(base_64_img, caption='Cropped Face')
      # Send the cropped face to visual search
      with st.spinner("Searching for this person's Profile"):
        vis_search_req = make_visual_search_req(base_64_img)
        urls = []
        scores = []
        names = []
        image_df = pd.DataFrame(columns=['url', 'input_type', 'score', 'metadata'])
        video_df = pd.DataFrame(columns=['url', 'input_type','file_name', 'frame_time',
                                         'frame_index', 'score', 'metadata'])
        for i in range(len(vis_search_req.hits)):
            if 'image' in MessageToDict(vis_search_req.hits[i].input.data):
                img_url = vis_search_req.hits[i].input.data.image.url
                input_type = 'image'
                frame_time_string = '-'

                score = vis_search_req.hits[i].score
                metadata = vis_search_req.hits[i].input.data.metadata

                name = ''
                if "id" in metadata.keys():
                    name = metadata["id"]
                # elif 'img_name':
                #     name = metadata['img_name'] 
                elif 'filename' in metadata.keys():
                    name = str(metadata['filename'])
                elif "person_name" in metadata.keys():
                    name = str(metadata["person_name"])
                else:
                    name = metadata

                image_df.loc[len(image_df)] = {'url': img_url, 'input_type': input_type, 
                                              'score': score, 'metadata': name}#+"\n"+file_name}
            else:
                img_url = vis_search_req.hits[i].input.data.video.url
                
                video_frames = vis_search_req.hits[i].annotation.data.frames
                delimiter = ":"
                frame_time_list = []

                for frame in video_frames:
                    # Convert frame time to seconds
                    frame_time = (frame.frame_info.time - 500) / 1000
                    frame_index = frame.frame_info.index
                    # Convert seconds to minutes and seconds
                    minutes, seconds = divmod(frame_time, 60)
                    # Format the time as mm:ss
                    frame_time_formatted = f"{int(minutes):02d}:{int(seconds):02d}"
                    frame_time_list.append(frame_time_formatted)

                frame_time_string = delimiter.join(frame_time_list)
                input_type = 'video'

                score = vis_search_req.hits[i].score
                metadata = vis_search_req.hits[i].input.data.metadata
                # st.write(metadata)
                
                file_name = metadata['filename']
                if "id" in metadata.keys():
                    name = metadata["id"]
                # elif 'img_name':
                #     name = metadata['img_name'] 
                # elif 'filename' in metadata.keys():
                #     file_name = metadata['filename']
                else:
                    name = metadata

                video_df.loc[len(video_df)] = {'url': img_url, 'input_type': input_type, 'frame_time': frame_time_string, 
                                                'frame_index': frame_index,'score': score, 'metadata': name, 'file_name': file_name}

        filtered_images = image_df[image_df['score'] > confidence_threshold].drop_duplicates(subset='score')
        video_df['frame_time_seconds'] = video_df['frame_time'].apply(convert_to_seconds)
        filtered_videos = video_df[video_df['score'] > confidence_threshold]
        # Drop duplicates based on 'file_name', 'frame_index', and 'score'
        filtered_videos = filtered_videos.drop_duplicates(subset=['file_name', 'frame_index', 'score'])
        st.dataframe(filtered_images)
        # st.dataframe(filtered_videos)

        # Display the images from filtered_images
        st.write('-' * 100)
        st.header('Images found with person in it')

        num_images = len(filtered_images)
        # Specify image width for uniformity
        image_width = 300

        # Iterate through the images and display them in a grid with 2 columns
        for i in range(0, num_images, 2):
          cols = st.columns(2)  # Create two columns
          # Display image in the first column (if available)
          if i < num_images:
              with cols[0]:
                  st.image(filtered_images.iloc[i]['url'], width=image_width, caption=filtered_images.iloc[i]['metadata'])
          # Display image in the second column (if available)
          if i + 1 < num_images:
              with cols[1]:
                  st.image(filtered_images.iloc[i + 1]['url'], width=image_width, caption=filtered_images.iloc[i + 1]['metadata'])

        plt.tight_layout()
        plt.show()
        st.write('-' * 100)
        st.header('Videos found with person in it')

        # Video data processing
        num_videos = len(filtered_videos)
        # Specify video width for uniformity
        video_width = 300
        # Convert 'metadata' column values to strings
        filtered_videos['metadata'] = filtered_videos['metadata'].astype(str)
        grouped_videos = filtered_videos.groupby(['metadata', 'file_name']).agg({
          'frame_index': lambda x: sorted(list(x)),  # Aggregate frame_index values into a list
          'score': ['count', 'min', 'max']  # Count, min, and max for scores
        }).reset_index()

        # Rename columns for clarity
        grouped_videos.columns = ['metadata', 'file_name', 'frame_indices', 'frame_count', 'min_score', 'max_score']

        # Display the grouped videos
        st.dataframe(grouped_videos)
        # Set to keep track of displayed video file names
        displayed_video_files = set()

        # Iterate through filtered_videos
        for _, video_row in filtered_videos.iterrows():
          print(video_row)
          file_name = video_row['file_name']
          file_url = video_row['url']
          frame_indices = video_row['frame_index']
          metadata = video_row['metadata']
          
          # Check if the video file name has already been displayed
          if file_name not in displayed_video_files:
            st.write(f"**Metadata**: {metadata} | **Frame Indices**: {frame_indices} | **File URL**: {file_url}")
            # Convert frame_indices to a list
            frame_indices = [frame_indices] if isinstance(frame_indices, int) else frame_indices
            # Display each unique frame from the video
            for frame_index in frame_indices:
              frame_url = f"{file_url}#t={frame_index}"
              st.video(frame_url, format='mp4', start_time=frame_index)
            # Add the video file name to the set to avoid repeating
            displayed_video_files.add(file_name)
        