# Continuous Machine Learning project integration with DVC

![img](image/README/1626973978814.png)**[Data Version Control](https://github.com/iterative/dvc)** or **DVC** is an **open-source** tool for data science and machine learning projects. It allows for different versioning and management of datasets and Machine Learning models.

Using github actions we can actually generate ML results for every Pull request and have the info we need right there,

### DVC Command Cheat Sheet


```
git init
dvc init
git commit -m "initial commit"
```

#set dvc remote

```
dvc remote add -d myremote gdrive://0AIac4JZqHhKmUk9PDA/dvcstore
git commit -m "sets dvc remote"
```

#process which repeats after any modification of data(new version)

---

#adds file to dvc and .gitignore

```
dvc add path_to_data
git add .gitignore && path_to_data.dvc
git commit -m "data:track"
```

#tags data version on git

```
git tag -a 'v1' -m "raw data"
dvc push
```

---

go ahead and delete the data!
it might also appear in .dvc/cache

to get data back => dvc pull
