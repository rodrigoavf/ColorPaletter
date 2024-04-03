# Color Palette Generator
This project is a Color Palette Generator built using Streamlit, a Python library for creating interactive web applications. The Color Palette Generator utilizes K-Means clustering algorithm to extract dominant colors from an image and display them along with a scatter plot showing the distribution of colors in the image. Additionally, it provides an option to repaint the image using the generated color palette.

## Usage
Clone the repository:

Copy code
git clone https://github.com/your_username/Color-Palette-Generator.git
Navigate to the project directory:

bash
Copy code
cd Color-Palette-Generator
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy code
streamlit run app.py
Features
Upload Image: Users can upload an image from their local system or provide a URL of the image.
Select Number of Colors (k): Users can choose the number of dominant colors they want in the generated color palette.
Generate Color Palette: Upon clicking the button, the application generates the color palette based on the selected image and number of colors.
Display Color Palette: The application displays the top k dominant colors along with their hexadecimal codes.
K-Means Clustering Visualization: It shows a scatter plot visualizing the distribution of colors in the image using K-Means clustering.
Repainted Image: Users can view the image repainted using the generated color palette.
Technologies Used
Streamlit: Used for creating the interactive web application.
NumPy: Utilized for numerical computing and array manipulation.
PIL (Python Imaging Library): Used for image processing tasks.
Scikit-learn: Employed for implementing the K-Means clustering algorithm.
Matplotlib: Utilized for data visualization, including scatter plots.
Contributors
Your Name
Contributor's Name
License
This project is licensed under the MIT License - see the LICENSE file for details.
