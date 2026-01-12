# cgpu Setup Guide (Custom Credentials)

The default `cgpu` configuration uses a shared developer project which is currently restricted. To use `cgpu` correctly, you need to set up your own Google Cloud credentials.

## Step 1: Create a Google Cloud Project

1.  Go to the [Google Cloud Console](https://console.cloud.google.com/).
2.  Click the project selector (top left) and choose **New Project**.
3.  Name it `montage-ai-cgpu` (or anything you like) and click **Create**.

## Step 2: Configure OAuth Consent Screen

1.  Go to **APIs & Services > OAuth consent screen**.
2.  Choose **External** (available to any user with a Google account) and click **Create**.
    *   *Note: If you choose Internal, only users in your organization can use it.*
3.  **App Information:**
    *   App name: `cgpu`
    *   User support email: Your email
    *   Developer contact information: Your email
4.  Click **Save and Continue**.
5.  **Scopes:** Click **Add or Remove Scopes**.
    *   Manually add this scope: `https://www.googleapis.com/auth/colaboratory`
    *   Also select `.../auth/userinfo.email` and `.../auth/userinfo.profile`.
    *   Click **Update**, then **Save and Continue**.
6.  **Test Users:**
    *   Click **Add Users**.
    *   Enter your own Google email address.
    *   Click **Save and Continue**.

## Step 3: Create Credentials

1.  Go to **APIs & Services > Credentials**.
2.  Click **Create Credentials** > **OAuth client ID**.
3.  **Application type:** Select **Desktop app**.
4.  **Name:** `cgpu-client`
5.  Click **Create**.
6.  **Copy** the **Client ID** and **Client Secret**.

## Step 4: Configure cgpu

Provide the Client ID and Client Secret to the assistant, or create the config file manually at `~/.config/cgpu/config.json`:

```json
{
  "clientId": "YOUR_CLIENT_ID",
  "clientSecret": "YOUR_CLIENT_SECRET",
  "colabApiDomain": "https://colab.research.google.com",
  "colabGapiDomain": "https://colab.googleapis.com"
}
```
