"""
WebSocket handler for real-time voice agent streaming
"""
import json
import base64
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional

from app.pipeline.voice_pipeline import VoiceAgentPipeline
from app.utils.logger import log


class WebSocketHandler:
    """Handle WebSocket connections for real-time voice processing"""
    
    def __init__(self, pipeline: VoiceAgentPipeline):
        """
        Initialize WebSocket handler
        
        Args:
            pipeline: Voice agent pipeline instance
        """
        self.pipeline = pipeline
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        log.info(f"WebSocket connected. Active connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        log.info(f"WebSocket disconnected. Active connections: {len(self.active_connections)}")
    
    async def handle_voice_stream(self, websocket: WebSocket):
        """
        Handle voice streaming through WebSocket
        
        Expected message format:
        {
            "type": "audio",
            "data": "base64_audio_data",
            "language": "en",
            "company_id": "techstore"
        }
        """
        await self.connect(websocket)
        
        try:
            # Send welcome message
            await websocket.send_json({
                "type": "connected",
                "message": "Voice agent ready"
            })
            
            while True:
                # Receive message
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "audio":
                    # Process audio through pipeline
                    try:
                        # Decode audio
                        audio_bytes = base64.b64decode(data["data"])
                        language = data.get("language", "en")
                        company_id = data.get("company_id")

                        # Send processing status
                        await websocket.send_json({
                            "type": "processing",
                            "stage": "transcribing"
                        })

                        # Process through pipeline
                        result = self.pipeline.process_audio(
                            audio_input=audio_bytes,
                            language=language,
                            company_id=company_id
                        )
                        
                        # Send transcription
                        await websocket.send_json({
                            "type": "transcription",
                            "text": result["transcription"]
                        })
                        
                        # Send LLM response
                        await websocket.send_json({
                            "type": "llm_response",
                            "text": result["llm_response"]
                        })
                        
                        # Send audio response
                        await websocket.send_json({
                            "type": "audio_response",
                            "data": result["audio_base64"],
                            "timing": result["timing"]
                        })
                        
                    except Exception as e:
                        log.error(f"Error processing audio: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                
                elif data.get("type") == "reset":
                    # Reset conversation
                    self.pipeline.reset_conversation()
                    await websocket.send_json({
                        "type": "reset_complete",
                        "message": "Conversation reset"
                    })
                
                elif data.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong"
                    })
                
        except WebSocketDisconnect:
            self.disconnect(websocket)
            log.info("Client disconnected")
        except Exception as e:
            log.error(f"WebSocket error: {e}")
            self.disconnect(websocket)