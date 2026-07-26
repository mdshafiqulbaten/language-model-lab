# Upload to GitHub

Intended repository:

`https://github.com/mdshafiqulbaten/language-model-lab`

## Browser method

1. Sign in to GitHub as `md2015`.
2. Create a new public repository named `language-model-lab`.
3. Do not add another README, license, or `.gitignore`.
4. Extract this ZIP.
5. Upload the contents of the `language-model-lab` folder, not the outer ZIP.
6. Commit with the message `Publish book companion code v1.0.0`.
7. Open the Actions tab and confirm the test workflow passes.
8. Create a release tagged `v1.0.0`.

## Terminal method

```bash
cd language-model-lab
git init
git branch -M main
git add .
git commit -m "Publish book companion code v1.0.0"
git remote add origin https://github.com/mdshafiqulbaten/language-model-lab.git
git push -u origin main
```

After the push:

```bash
git tag -a v1.0.0 -m "Book companion release v1.0.0"
git push origin v1.0.0
```

