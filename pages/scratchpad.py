image_df = pd.DataFrame(columns=['url', 'input_type', 'score', 'metadata'])
video_df = pd.DataFrame(columns=['url', 'input_type', 'frame', 'score', 'metadata'])

for face in detected_faces:            
    # Save the cropped face as a base64 string
    st.write('-' * 100)
    base_64_img = face.data.image.base64
    st.image(base_64_img, caption='Cropped Face')

    # Send the cropped face to visual search
    with st.spinner("Searching for this face"):
        vis_search_req = make_visual_search_req(base_64_img)

        for i in range(len(vis_search_req.hits)):
            if 'image' in MessageToDict(vis_search_req.hits[i].input.data):
                img_url = vis_search_req.hits[i].input.data.image.url
                input_type = 'image'
                frame_time_string = '-'

                score = vis_search_req.hits[i].score
                metadata = vis_search_req.hits[i].input.data.metadata

                if 'img_name' in metadata.keys():
                    name = metadata['img_name']
                elif 'filename' in metadata.keys():
                    name = metadata['filename']
                else:
                    name = metadata

                image_df.loc[len(image_df)] = {'url': img_url, 'input_type': input_type, 
                                               'score': score, 'metadata': name}

            else:
                img_url = vis_search_req.hits[i].input.data.video.url
                
                video_frames = vis_search_req.hits[i].annotation.data.frames
                delimiter = ":"
                frame_time_list = []

                for frame in video_frames:
                    # Convert frame time to seconds
                    frame_time = (frame.frame_info.time - 500) / 1000
                    # Convert seconds to minutes and seconds
                    minutes, seconds = divmod(frame_time, 60)
                    # Format the time as mm:ss
                    frame_time_formatted = f"{int(minutes):02d}:{int(seconds):02d}"
                    frame_time_list.append(frame_time_formatted)

                frame_time_string = delimiter.join(frame_time_list)
                input_type = 'video'

                score = vis_search_req.hits[i].score
                metadata = vis_search_req.hits[i].input.data.metadata

                if 'img_name' in metadata.keys():
                    name = metadata['img_name']
                elif 'filename' in metadata.keys():
                    name = metadata['filename']
                else:
                    name = metadata

                video_df.loc[len(video_df)] = {'url': img_url, 'input_type': input_type, 'frame': frame_time_string, 
                                               'score': score, 'metadata': name}
