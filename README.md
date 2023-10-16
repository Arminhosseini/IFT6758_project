# IFT6758 Project
[Repo URL](https://github.com/Arminhosseini/IFT6758_project)

## Authors
- [@Shilong Zhou](https://github.com/sl-Zhou)
- [@Seyed Armin Hosseini](https://github.com/Arminhosseini)
- [@Sepehr Babapour](https://github.com/sprbpr)
- [@Thai Vu Nguyen](https://github.com/NguyenThaiVu)

## Blog post page
[Click here!](https://nguyenthaivu.github.io/)

## Prerequisites
- pandas
- numpy
- matplotlib
- seaborn
- requests
- scipy
- plotly
- ipywidgets
  
Of course the python environment is essential!

## Run Locally

### Step 1. Clone the project
```
  git clone https://github.com/Arminhosseini/IFT6758_project.git
```
Or you can just download our zip file!

### Step 2. Install Environment
```
cd IFT6758_project
virtualenv venv
source venv/bin/activate
pip install -r requirement.txt
```

### Step 3. Download NHL Play-by-Play Data (It will take some time :))
``` 
python crawler.py
```
You can also check out [this page](https://github.com/Arminhosseini/IFT6758_project/blob/main/docs/crawler.md) for a more detailed guide

### Step 4. Check the Interactive NHL Game Data Panel
```
run interactive_debugging_tool.ipynb
```
You can also check out [this page](https://github.com/Arminhosseini/IFT6758_project/blob/main/docs/interactive_debugging_tool.md) for a more detailed guide

### Step 5. Clean data to get the tidy data
```
python tidyData.py
```

### Step 6. Check the simple visualization
```
python simple_visualization.py
```

### Step 7. Check the advanced visualization
```
python advanced_visualization.py
```

### Please note that all our images are stored in the folder [images](https://github.com/Arminhosseini/IFT6758_project/tree/main/images)


