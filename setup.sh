mkdir -p /app/.streamlit/
echo "\
[server]
port = $PORT
enableCORS = false
headless = true

" > /app/.streamlit/config.toml
