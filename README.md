# Dash Application Setup and Usage

This guide will help you install Dash, run the `app.py` file, and open the visualization in your browser.

## Requirements

- Python 3.6 or higher
- `pip` (Python package manager)
- pandas
- plotly
- numpy
- matplotlib
## Step 1: Install Dash

1. Open your terminal or command prompt.
2. Install Dash by running the following command:

   ```bash
   pip install dash
This command will install Dash and its required dependencies. We assumed you have all the other libraries mentioned in requirements.


## Step 2: Run the app.py file
After Dash is installed, navigate to the folder containing your app.py file using the terminal or command prompt. For example:

   ```bash
   cd /path/to/your/app
Once you're in the folder with app.py, run the app using the following command:

   ```bash
   python dash_server.py
This will start the Dash application, and you'll see output in the terminal that looks like:

Dash is running on http://127.0.0.1:8050/
The application is now running locally on the IP address 127.0.0.1 (also known as localhost) and port 8050 by default.

## Step 3: Open the app in your browser
Open your web browser (e.g., Chrome, Firefox).
In the address bar, type the IP and port shown in the terminal (e.g., http://127.0.0.1:8050/) and press Enter.
If you're running the app locally, the IP will likely be 127.0.0.1 or localhost, and the port will be 8050 by default.
If the terminal shows a different IP or port, use that instead.
## Step 4: Start using the visualization
Once the app is open in your browser, you can interact with the visualization and explore the features provided by the Dash app.
