import pandas as pd
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from google.protobuf import json_format
import numpy as np
import requests
from google.protobuf.json_format import MessageToDict
from PIL import Image
from io import BytesIO
from clarifai.modules.css import ClarifaiStreamlitCSS
import pandas as pd

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
userDataClarifaiMain= resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')
#request headers
headers = {'Authorization': f'Key {auth._pat}'}
# st.write(headers)

def list_1000_inputs():
    list_inputs_response = stub.ListInputs(
    service_pb2.ListInputsRequest(
        user_app_id=userDataObject,
        page=1, 
        per_page=1000
    ),
    metadata=(('authorization', 'Key ' + auth._pat),)
    )
    return list_inputs_response

def display_images_in_rows(df, header_title, images_per_row=2, standard_height=200, spacer_width=1):
    """
    Displays images or videos in rows with a specified number of items per row.
    Each image is resized to have a consistent height, maintaining its aspect ratio.
    Videos are displayed using Streamlit's st.video function.
    Spacing is added between columns.
    """
    if not df.empty:
        st.header(header_title)
        total_rows = len(df) // images_per_row + (len(df) % images_per_row > 0)

        for row_num in range(total_rows):
            start_index = row_num * images_per_row
            end_index = start_index + images_per_row
            row_df = df.iloc[start_index:end_index]

            # Adjust the number of columns to include spacers
            cols = st.columns([1]*images_per_row + [spacer_width]*(images_per_row-1))

            # Iterate over columns with content, skipping spacer columns
            col_index = 0
            for _, image_data in row_df.iterrows():
                col = cols[col_index]
                col_index += 1 + spacer_width  # Skip the spacer columns
                url = image_data['url']
                filename = image_data['filename']

                if url:
                    if filename.lower().endswith('.mp4'):
                        # Display video
                        col.video(url)
                    else:
                        # Display image
                        response = requests.get(url, headers=headers, stream=True)
                        img = Image.open(BytesIO(response.content))

                        # Calculate new width to maintain aspect ratio
                        aspect_ratio = img.width / img.height
                        new_width = int(standard_height * aspect_ratio)

                        # Resize the image
                        img = img.resize((new_width, standard_height))

                        # Convert the image object back to bytes for Streamlit to display
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        byte_im = buf.getvalue()

                        # Truncate filename to 10 characters if longer
                        truncated_filename = (filename[:10] + '...') if len(filename) > 10 else filename

                        caption = f"Filename: {truncated_filename}, \n\n Person ID: {image_data['person_id']}"
                        col.image(byte_im, caption=caption, use_column_width=False)

            if row_num < total_rows - 1:
                st.write("")  # Add space after each row except the last one



if __name__ == "__main__":
    ClarifaiStreamlitCSS.insert_default_css(st)
    st.header("Clarifai Profile Viewer")
     
    # Get the list of inputs from the Clarifai API
    list_inputs_response = list_1000_inputs()
    # print(list_inputs_response)

    # Create a dropdown with a default '-' option followed by options from 'Person-01' to 'Person-14' and then from 'Person-21' to 'Person-57'
    person_choices = ['-', *[f"Person-{str(i).zfill(2)}" for i in range(1, 15)], *[f"Person-{str(i).zfill(2)}" for i in range(21, 58)]]
    selected_person = st.selectbox("Select a person:", person_choices, index=0)

    if selected_person:
        df = pd.DataFrame(columns=['input_id', 'url', 'metadata','filename', 'person_id', 'confidence'])
        for i in range(len(list_inputs_response.inputs)):
            # st.write(list_inputs_response.inputs[i].data)
            if 'image' in MessageToDict(list_inputs_response.inputs[i].data):
                img_url = list_inputs_response.inputs[i].data.image.url
            else:
                img_url = list_inputs_response.inputs[i].data.video.url
            # st.write(img_url)
            metadata = list_inputs_response.inputs[i].data.metadata
            if "id" in metadata.keys() and metadata['id'] == selected_person :
                person_id = metadata['id']
                file_name = metadata['filename']
                print("id : ",metadata["id"])
            
                # Confidence, if it exists, then take that value else treat it as high
                if 'confidence' in metadata.keys():
                    confidence = metadata['confidence']
                else: 
                    confidence = 'high'
                df.loc[len(df)] = {'input_id': list_inputs_response.inputs[i].id,
                                    'url': img_url,
                                    'metadata': metadata,
                                    'person_id': person_id,
                                    'filename':file_name,
                                    'confidence': confidence}
        # Sort the dataframe by the confidence column
        sorted_df = df.sort_values(by='confidence')

        # Add collapsible container for the dataframe
        with st.expander("Sorted Dataframe", expanded=False):
            st.write(sorted_df)

        # Create separate dataframes based on confidence values
        high_confidence_df = sorted_df[sorted_df['confidence'] == 'high']
        low_confidence_df = sorted_df[sorted_df['confidence'] == 'low']

        # Display images with high confidence
        display_images_in_rows(high_confidence_df, "Images with High Confidence", images_per_row=2)

        # Display images with low confidence
        display_images_in_rows(low_confidence_df, "Images with Low Confidence", images_per_row=2)

       