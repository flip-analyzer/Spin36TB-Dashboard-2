# 🔌 OANDA Setup Instructions

## 🚀 Step 1: Get Your OANDA API Credentials

### Go to Your OANDA Account:
1. **Login** to your OANDA account
2. **Navigate to**: Account → Manage API Access (or Manage Funds → API)
3. **Generate Personal Access Token**
   - Click "Generate Token" 
   - Copy the token (looks like: `b3a4d5f6g7h8i9j0k1l2m3n4o5p6q7r8`)
4. **Find Your Account ID**
   - Should be visible on the same page
   - Format: `123-456-7890123-001`

## 🔧 Step 2: Configure the Downloader

### Option A: Edit the File Directly
1. Open `oanda_data_downloader.py`
2. Find these lines (around line 12):
```python
OANDA_CONFIG = {
    'api_token': 'YOUR_API_TOKEN_HERE',  # Replace this
    'account_id': 'YOUR_ACCOUNT_ID_HERE',  # Replace this  
    'environment': 'practice',
}
```
3. **Replace with your actual credentials**:
```python
OANDA_CONFIG = {
    'api_token': 'b3a4d5f6g7h8i9j0k1l2m3n4o5p6q7r8',  # Your token
    'account_id': '123-456-7890123-001',  # Your account ID
    'environment': 'practice',
}
```

### Option B: Create Credentials File
Create a new file called `oanda_credentials.py`:
```python
# OANDA Credentials - Keep this file private!
API_TOKEN = "your_token_here"
ACCOUNT_ID = "your_account_id_here"
ENVIRONMENT = "practice"  # or "live"
```

## 🏃‍♂️ Step 3: Run the Download

```bash
cd /Users/jonspinogatti/Desktop/spin36TB
python oanda_data_downloader.py
```

### Expected Output:
```
✅ OANDA connection successful!
   Currency: USD
   Balance: 100000.00

📊 Downloading EUR_USD data...
   Granularity: M5 (5-minute bars)
   Count: 35000 candles

✅ Downloaded 34,567 complete candles
   Date range: 2024-03-15 to 2024-08-30
   Price range: 1.0649 - 1.1139

💾 Data saved to: oanda_eurusd_6months.csv
```

## 🎯 Step 4: Validate Against Real Data

Once you have the CSV file, run:
```bash
python historical_validation.py
```

## 🔒 Security Notes:

- **Keep your API token private** - don't share it
- **Use practice environment** initially (no real money risk)
- **Regenerate token** if you think it's compromised
- **Never commit credentials** to version control

## 🆘 Troubleshooting:

### "Connection Failed" Error:
- Check your API token is correct
- Verify account ID format
- Make sure you're using practice environment initially
- Check OANDA account is active

### "Invalid Instrument" Error:
- Use 'EUR_USD' (not 'EURUSD')
- Check instrument is available in your account

### "Rate Limit" Error:
- OANDA limits API calls
- Wait a few minutes and try again
- Reduce the count parameter if needed

## ✅ Success Criteria:

You'll know it worked when you see:
- ✅ Connection successful message
- ✅ Downloaded thousands of candles
- ✅ CSV file created in your folder
- ✅ Price range looks realistic (1.06-1.11)

**Ready for the next step: Historical validation!** 🚀