import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import pandas as pd
import plotly.graph_objects as go

def get_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error("Error loading image from URL. Please make sure the URL is correct.")
        st.stop()

def resize_image(image, base_width=500):
    # Calculate the aspect ratio
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    # Resize the image
    img = image.resize((base_width, h_size), Image.LANCZOS)
    return img

def get_colors_from_image(image, k):
    # Resize image for faster processing
    img = resize_image(image)
    # Convert image to numpy array
    img_array = np.array(img)
    # Reshape the array to make it suitable for KMeans
    img_array = img_array.reshape((-1, 3))
    # Fit KMeans model on the image
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(img_array)
    # Get the cluster centers which represent the dominant colors
    colors = kmeans.cluster_centers_
    # Get labels for each pixel
    labels = kmeans.labels_
    return colors.astype(int), labels

def display_color_palette(colors):
    pallet_table = {}
    palette = np.zeros((100, len(colors)*100, 3), dtype=int)
    plt.figure(figsize=(8, 2))
    for i, color in enumerate(colors):
        hex_code = '#%02x%02x%02x' % tuple(color)
        palette[:, i*100:(i+1)*100, :] = color
        plt.text(i * 100 + 50, 50, hex_code, ha='center', va='center', fontsize=8, color='black')
        pallet_table[f"Cluster {i+1}"] = [hex_code]
    plt.imshow(palette)
    plt.axis('off')
    st.pyplot(plt)
    df = pd.DataFrame(pallet_table)
    st.dataframe(data=df, use_container_width=True, hide_index=True)

def display_scatter_plot(image, labels, colors, option):
    if option == "3D Interactive Plot":
        # Resize image for faster processing
        img = resize_image(image)
        # Convert image to numpy array
        img_array = np.array(img)
        # Reshape the array to make it suitable for plotting
        img_array = img_array.reshape((-1, 3))
        
        # Create trace for each cluster
        traces = []
        for i, color in enumerate(colors):
            cluster_points = img_array[labels == i]
            trace = go.Scatter3d(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                z=cluster_points[:, 2],
                mode='markers',
                marker=dict(
                    color=f'rgb({color[0]}, {color[1]}, {color[2]})',
                    size=5,
                    opacity=0.8
                ),
                name=f'Cluster {i+1}'
            )
            traces.append(trace)

        # Create layout for the plot
        layout = go.Layout(
            scene=dict(
                xaxis=dict(title='R'),
                yaxis=dict(title='G'),
                zaxis=dict(title='B')
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(orientation="h")
        )

        # Create figure
        fig = go.Figure(data=traces, layout=layout)
        
        # Display the plot
        st.plotly_chart(fig)
    if option == "2D Simple Plot":
        # Resize image for faster processing
        img = resize_image(image)
        # Convert image to numpy array
        img_array = np.array(img)
        # Reshape the array to make it suitable for plotting
        img_array = img_array.reshape((-1, 3))
        # Plot scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, color in enumerate(colors):
            cluster_points = img_array[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color / 255], label=f'Cluster {i+1}')
        ax.set_title('K-Means Clustering')
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.legend()
        st.pyplot(fig)
    if option == "3D Simple Plot":
        # Resize image for faster processing
        img = resize_image(image)
        # Convert image to numpy array
        img_array = np.array(img)
        # Reshape the array to make it suitable for plotting
        img_array = img_array.reshape((-1, 3))
        # Plot scatter plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        for i, color in enumerate(colors):
            cluster_points = img_array[labels == i]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color / 255], label=f'Cluster {i+1}')
        ax.set_title('K-Means Clustering')
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        ax.legend()
        st.pyplot(fig)

def repaint_image(image, labels, colors):
    # Resize image for faster processing
    img = resize_image(image)
    # Convert image to numpy array
    img_array = np.array(img)
    # Reshape the array to make it suitable for repainting
    img_array = img_array.reshape((-1, 3))
    # Repaint the image
    for i in range(len(colors)):
        img_array[labels == i] = colors[i]
    # Reshape the array back to the original image shape
    repainted_img_array = img_array.reshape(img.size[::-1] + (3,))
    # Convert numpy array to image
    repainted_img = Image.fromarray(repainted_img_array.astype('uint8'), 'RGB')
    return repainted_img

def main():
    st.set_page_config(page_title="Color Paletter",
                       page_icon='🎨')
    st.session_state.setdefault("uploaded_file", None)

    st.title("Color Paletter")
    st.write("A color palette generator")
    st.write("## Settings")

    if st.session_state["uploaded_file"] == None:
        k = st.slider("Number of colors (k)", min_value=2, max_value=10, value=5)
        upload_option = st.radio("Upload image from", ("Upload", "URL"))
        if upload_option == "Upload":
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            url = st.text_input("Enter Image URL")
            if url:
                image = get_image_from_url(url)
                st.image(image, caption="Image from URL", use_column_width=True)

    if image is not None:
        try:
            colors, labels = get_colors_from_image(image, k)
            st.subheader(f"Top {k} Colors:")
            display_color_palette(colors)
            st.subheader("K-Means Clustering:")
            option = st.selectbox(label="Plot type", options=["2D Simple Plot", "3D Simple Plot", "3D Interactive Plot"])
            display_scatter_plot(image, labels, colors, option)
            st.subheader("Repainted Image:")
            repainted_image = repaint_image(image, labels, colors)
            st.image(repainted_image, caption="Repainted image using the color pallet", use_column_width=True)
        except Exception as e:
            st.error("An error occurred while generating the color palette.")
            st.error(e)

if __name__ == "__main__":
    main()
