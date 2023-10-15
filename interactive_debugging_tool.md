##### Table of Contents
[Example-Usage](#example-usage)
[Emphasis](#emphasis)


## Guide to Interactive NHL Game Data Panel

In this guide, we'll explore how to use an interactive Python script to access and visualize NHL game data. The script includes graphical user interface elements to select, view, and plot game data, and it utilizes the `Crawler` class to fetch NHL data.

### Prerequisites

Before you get started, make sure you have the following prerequisites in place:

- **Python:** Make sure you have Python installed on your system. You can download it from python.org.

- **Jupyter Notebook:** You should be running this script within a Jupyter Notebook environment.

- **Required Python Packages:** You will need the following Python packages, which can be installed using pip:

```bash script
pip install ipywidgets matplotlib pillow
```

### Using the Interactive Panel

The interactive panel allows you to explore NHL game data in a user-friendly manner. Let's break down how to use each part of the code:

#### Importing Necessary Libraries and Modules

This block imports essential Python modules, such as IPython widgets for creating the GUI, Matplotlib for data visualization, Pillow for image manipulation, and NumPy for data manipulation. It also imports the `Crawler` class from an external module.

#### Creating an Instance of the 'Crawler' Class

In this block, an instance of the `Crawler` class is created. The `Crawler` class handles the retrieval of NHL game data.

#### Defining the `Panel` Class

The `Panel` class is introduced to manage the graphical user interface for data selection and display. The following functionalities are implemented:

- **Season Selection:** You can select your desired season among all available seasons automatically specified by the `crawler`. The season is also updated based on the entered game ID.

- **Game Type Selection:** You can select your desired game type between regular and playoff games. The game type is also updated based on the entered game ID.

- **Game ID Selection:** The game ID is adjusted based on the selected season and game type. The only thing you may want to change is the game ID (last 4 digits).

- **'Go' Button:** This button fetches and displays selected game data, allowing you to explore the data interactively.

- **'Show' Button:** This button displays the selected data in a user-friendly format.

- **'Reset' Button:** This button resets the control panel to its default values.

- **Data Selection:** It enables you to select data categories and drill down into the information.


#### Creating an Instance of the `Panel` Class

In this block, an instance of the `Panel` class is created, which initiates the interactive panel in your Jupyter Notebook environment.

### Example Usage

- Open a Jupyter Notebook environment.

- Copy and paste the provided code blocks into separate cells in your notebook.

- Run each code block sequentially.

- Once you run the last code block, the interactive panel will be displayed in your notebook.

- You can start by selecting the season and game type, and then click the 'Go' button to fetch and display NHL game data.

- Use the 'Show' button to view the selected data in a readable format. This is an example of debugging tool visualization:

![](interactive_debugging_tool_visualization.png)

- You can also reset the panel using the 'Reset' button to start fresh.

- The panel allows you to explore and visualize data for specific NHL games.

The interactive panel is a powerful tool for accessing and visualizing NHL game data with ease. Enjoy exploring and analyzing NHL game data interactively!