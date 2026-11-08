# Road Accident Intelligence — Public Website

This Streamlit application loads its project datasets automatically from the `data/` folder. Visitors do not need to upload CSV files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy publicly with Streamlit Community Cloud

1. Create a GitHub repository (for example, `road-accident-intelligence`).
2. Upload `app.py`, `requirements.txt`, and the complete `data/` folder to the repository.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Select **Create app**, choose your repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will provide a public URL that can be shared with teachers and other users.

## Updating the stored dataset

Replace the relevant CSV in `data/` in GitHub and commit the change. The app will redeploy and use the updated file. Keep the filenames unchanged unless you also update `app.py`.

## Important notes

- The included data is bundled with the app, so it is suitable for a public, read-only academic dashboard.
- Anyone who can access the public URL can view the dataset-derived information. Do not commit personal, confidential, or sensitive data.
- This setup does not let visitors permanently submit new datasets. If you later want user accounts, visitor uploads, and saved records, use a database/object-storage service with authentication and access controls.
- Prediction output is educational and should not be used for safety-critical decisions.
