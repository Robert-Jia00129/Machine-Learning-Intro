# This folder shows necessary steps to set up Jupyter Notebook on Mac


## Downloading Anaconda or VS Code
### Anaconda: You can download [anaconda-naviagtor](https://www.anaconda.com/products/distribution) online. It is pretty straight forward.
After downloading, I reference [this video](https://www.youtube.com/watch?v=tGkZ9EARwzk) for downloading tensorflow environment

---

### VS Code: If you want to download VS Code, and are unable to download: 
Try this link: https://vscode.cdn.azure.cn/stable/b4c1bd0a9b03c749ea011b06c6d2676c8091a70c/VSCode-darwin-universal.zip 

---

## Importing Necessary Libraries

### Tensorflow
Basically, the code you put in your terminal is: 
```
conda create -n tf tensorflow
```
Else, you could directly type the command in jupyter notebook
```python
!pip -qq install tensorflow
```

And the library would be downloaded so you can import it into your jupyter notebook. 

## Useful Links to refer to: 
[Setting up Jupyter Notebook on M1](https://blog.roboflow.com/how-to-run-jupyter-notebooks-on-a-mac-m1/)

[Jupyter Notebook Tips and Tricks](https://towardsdatascience.com/15-tips-and-tricks-for-jupyter-notebook-that-will-ease-your-coding-experience-e469207ac95c)

I faced difficulties opening the terminal, if you have the same problem, just click the play button here
![Example Imiage](https://raw.githubusercontent.com/Z-Robert-Jia/Machine-Learning-Intro/main/Jupyter%20Notebook%20Setupt/CondaSetup.png)

