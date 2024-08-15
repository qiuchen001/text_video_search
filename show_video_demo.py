import gradio as gr

def video_identity(video):
    return video

demo = gr.Interface(video_identity,
                    gr.Video(value=r"E:\workspace\work_data\videos\120266-720504932_small.mp4", label=None),
                    "playable_video",
                    )

if __name__ == "__main__":
    demo.launch()