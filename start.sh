#!/bin/bash
echo "🚀 啟動 AI Chatbot SaaS..."

# 啟動後端
echo "▶ 啟動後端 (port 8000)..."
cd backend
source venv/bin/activate
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!

# 啟動前端
echo "▶ 啟動前端 (port 3000)..."
cd ../frontend
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ 全部啟動完成！"
echo "   後端 API: http://localhost:8000"
echo "   前端介面: http://localhost:3000"
echo "   API 文件: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止全部服務"

wait
