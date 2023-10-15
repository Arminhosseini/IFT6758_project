## Download NHL Play-by-Play Data with Python

If you're looking to download NHL play-by-play data for analysis or other purposes, you're in the right place :)). This guide will walk you through using the `Crawler` class, a Python script that can help you scrape and store NHL game data, including play-by-play data.

### Prerequisites

Before you get started, ensure you have the following prerequisites:

- Python installed on your system.
- Libraries: `requests` and `pathlib`. You can install these with pip:

```bash script
pip install requests
```

### Getting Started
The Python script, `crawler.py`, will help you download NHL play-by-play data. Here's how you can use it:

1. Download the script from the provided location and save it to your local machine.

2. Open a command prompt or terminal window and navigate to the directory where you saved the script.

3. Run the script using the following command:

    ``` bash script
    python crawler.py
    ```

4. The script will start collecting NHL play-by-play data for the specified range of seasons. You can adjust the start and end seasons by modifying the `start_season` and `end_season` values in the script.

5. The collected data will be stored in a directory named `Dataset` within the script's directory.

### Explanation of the Python Script

Let's break down the key parts of the Python script:

#### Importing Necessary Modules

* The script starts by importing the required Python modules, including `Path` for file manipulation, `requests` for web requests, and `json` for handling JSON data.

#### Class Definition: `Crawler`

* The script defines a Python class named `Crawler`. This class encapsulates the functionality to fetch and store NHL game data. The script provides various methods within this class to handle data retrieval, storage, and access.

#### Data Storage and Initialization

* The class initializes instance variables such as the base URL for the NHL data API, paths for data storage, dictionaries for game types, and data storage containers.

#### Generating Game IDs

* The class provides methods to generate game IDs for regular season and playoff games, which are used to fetch game data from the NHL API.

#### Data Collection

* The `get_url` method constructs the full URL for fetching game data, and the `get_game_data` method fetches data from the NHL API using the constructed URL. The script checks the response status code to ensure a successful request.
* Also, the script defines functions to retrieve data for regular season and playoff games. These functions use the generated game IDs and fetch the corresponding game data.
* Finally, the `get_total_data` method collects data for a specified range of seasons, storing it in the `self.data` dictionary.

#### Data Storage

* The `write_data` method writes the collected game data to JSON files in the `Dataset` directory, organizing them by season and game type.

#### Dataset Information

* The `write_dataset_info` method writes information about the dataset, including the start and end seasons, to a JSON file.

#### Data Retrieval

* The `read_data` method reads all the collected data from JSON files and organizes it into a structured dictionary.
* The `read_data_by_game_id` method allows you to retrieve data for a specific game by providing its game ID.

### Example Usage

Here's an example of how to use the script to collect NHL play-by-play data:

``` bash script
    python crawler.py
```

This command will collect NHL play-by-play data for the specified range of seasons and store it in the "Dataset" directory. You can access the collected data and retrieve specific game data using the provided methods within the `Crawler` class.

With this script, you can easily download NHL play-by-play data and perform various analyses or build applications based on the obtained data. Enjoy exploring and working with NHL game data!