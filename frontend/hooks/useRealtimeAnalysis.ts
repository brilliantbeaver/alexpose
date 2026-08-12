/**
 * Realtime Analysis Hook
 * 
 * Custom hook for managing WebSocket connection and realtime pose analysis state.
 * Provides a clean interface for components to interact with the realtime service.
 */

'use client';

import { useState, useEffect, useCallback, useRef } from 'react';

interface PoseKeypoint {
  x: number;
  y: number;
  confidence: number;
  id: number;
}

interface PoseResult {
  keypoints: PoseKeypoint[];
  confidence_scores: number[];
  processing_time_ms: number;
  frame_id: number;
  timestamp: number;
  estimator_info: any;
}

interface GaitMetrics {
  cadence?: number | null;
  step_length?: number | null;
  stride_length?: number | null;
  walking_speed?: number | null;
  symmetry_index?: number | null;
  stability_score?: number | null;
  confidence: number;
  timestamp: number;
}

interface ProcessingStats {
  frames_received: number;
  frames_processed: number;
  frames_failed: number;
  average_processing_time_ms: number;
  poses_detected: number;
  gait_analyses_completed: number;
  session_duration_seconds: number;
}

interface RealtimeConfig {
  processing_mode: 'fast' | 'balanced' | 'accurate';
  buffer_size: number;
  enable_tracking: boolean;
  confidence_threshold: number;
  show_keypoints: boolean;
  show_skeleton: boolean;
  target_fps: number;
}

interface UseRealtimeAnalysisReturn {
  // Connection state
  isConnected: boolean;
  isProcessing: boolean;
  sessionId: string | null;
  error: string | null;
  
  // Data
  currentPose: PoseResult | null;
  gaitMetrics: GaitMetrics | null;
  statistics: ProcessingStats | null;
  config: RealtimeConfig;
  
  // Actions
  connect: () => Promise<void>;
  disconnect: () => void;
  sendFrame: (frameData: string) => void;
  updateConfig: (newConfig: Partial<RealtimeConfig>) => void;
  getStats: () => Promise<void>;
}

const DEFAULT_CONFIG: RealtimeConfig = {
  processing_mode: 'balanced',
  buffer_size: 30,
  enable_tracking: true,
  confidence_threshold: 0.5,
  show_keypoints: true,
  show_skeleton: true,
  target_fps: 25
};

export function useRealtimeAnalysis(): UseRealtimeAnalysisReturn {
  // Connection state
  const [isConnected, setIsConnected] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Data state
  const [currentPose, setCurrentPose] = useState<PoseResult | null>(null);
  const [gaitMetrics, setGaitMetrics] = useState<GaitMetrics | null>(null);
  const [statistics, setStatistics] = useState<ProcessingStats | null>(null);
  const [config, setConfig] = useState<RealtimeConfig>(DEFAULT_CONFIG);
  
  // WebSocket reference
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 5;
  
  // Frame throttling - only send new frame after receiving previous result
  const isProcessingFrameRef = useRef(false);
  const pendingFrameRef = useRef<string | null>(null);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  const connect = useCallback(async (): Promise<void> => {
    if (isConnected || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setError(null);
    
    try {
      // Determine WebSocket URL - use backend port directly
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      // Frontend runs on 3000, backend on 8000
      const wsUrl = `${protocol}//localhost:8000/api/realtime/stream`;
      
      console.log('Connecting to WebSocket:', wsUrl);
      
      // Create WebSocket connection
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      // Set up event handlers
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
        reconnectAttempts.current = 0;
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.onclose = (event) => {
        console.log('WebSocket closed:', event.code, event.reason);
        setIsConnected(false);
        setIsProcessing(false);
        wsRef.current = null;
        
        // Attempt reconnection if not a clean close
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          scheduleReconnect();
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error occurred');
      };

      // Wait for connection to open
      await new Promise<void>((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error('Connection timeout'));
        }, 10000);

        ws.onopen = () => {
          clearTimeout(timeout);
          setIsConnected(true);
          setError(null);
          reconnectAttempts.current = 0;
          resolve();
        };

        ws.onerror = () => {
          clearTimeout(timeout);
          reject(new Error('Connection failed'));
        };
      });

    } catch (error) {
      console.error('Failed to connect:', error);
      setError(error instanceof Error ? error.message : 'Connection failed');
      throw error;
    }
  }, [isConnected]);

  const disconnect = useCallback(() => {
    // Clear reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close(1000, 'User disconnected');
      wsRef.current = null;
    }

    // Reset state
    setIsConnected(false);
    setIsProcessing(false);
    setSessionId(null);
    setCurrentPose(null);
    setGaitMetrics(null);
    setStatistics(null);
    setError(null);
    reconnectAttempts.current = 0;
  }, []);

  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }

    const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
    reconnectAttempts.current++;

    console.log(`Scheduling reconnect attempt ${reconnectAttempts.current} in ${delay}ms`);
    
    reconnectTimeoutRef.current = setTimeout(() => {
      if (!isConnected) {
        connect().catch(error => {
          console.error('Reconnect failed:', error);
        });
      }
    }, delay);
  }, [isConnected, connect]);

  const sendFrame = useCallback((frameData: string) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    // If already processing a frame, store this one as pending
    if (isProcessingFrameRef.current) {
      pendingFrameRef.current = frameData;
      return;
    }

    try {
      const message = {
        type: 'frame',
        data: frameData.split(',')[1] // Remove data URL prefix
      };

      isProcessingFrameRef.current = true;
      wsRef.current.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to send frame:', error);
      isProcessingFrameRef.current = false;
    }
  }, []);

  const updateConfig = useCallback((newConfig: Partial<RealtimeConfig>) => {
    const updatedConfig = { ...config, ...newConfig };
    setConfig(updatedConfig);

    // Send config update to server
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      try {
        const message = {
          type: 'config_update',
          config: updatedConfig
        };

        wsRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send config update:', error);
      }
    }
  }, [config]);

  const getStats = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return;
    }

    try {
      const message = {
        type: 'get_stats'
      };

      wsRef.current.send(JSON.stringify(message));
    } catch (error) {
      console.error('Failed to request stats:', error);
    }
  }, []);

  const handleWebSocketMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'session_started':
        setSessionId(message.session_id);
        setIsProcessing(true);
        break;

      case 'pose_result':
        // Mark frame as processed
        isProcessingFrameRef.current = false;
        
        if (message.data?.success) {
          // Update pose data
          if (message.data.pose) {
            console.log('[DEBUG] Received pose with', message.data.pose.keypoints?.length || 0, 'keypoints');
            setCurrentPose(message.data.pose);
          } else {
            console.log('[DEBUG] No pose data in message');
          }

          // Update gait metrics if available
          if (message.data.gait_metrics) {
            setGaitMetrics(message.data.gait_metrics);
          }
        } else {
          console.log('[DEBUG] Pose result not successful:', message.data);
        }
        
        // Send pending frame if any
        if (pendingFrameRef.current) {
          const pendingFrame = pendingFrameRef.current;
          pendingFrameRef.current = null;
          sendFrame(pendingFrame);
        }
        break;

      case 'statistics':
        if (message.data) {
          setStatistics(message.data.processor_stats || message.data);
        }
        break;

      case 'config_updated':
        if (message.config) {
          setConfig(prevConfig => ({ ...prevConfig, ...message.config }));
        }
        break;

      case 'error':
        setError(message.message || 'Unknown error occurred');
        isProcessingFrameRef.current = false; // Reset on error
        break;

      case 'keepalive':
        // Respond to keepalive
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(JSON.stringify({ type: 'pong' }));
        }
        break;

      case 'pong':
        // Handle pong response
        break;

      default:
        break;
    }
  }, [sendFrame]);

  // Periodic stats request
  useEffect(() => {
    if (!isConnected || !isProcessing) {
      return;
    }

    const interval = setInterval(() => {
      getStats();
    }, 2000); // Request stats every 2 seconds

    return () => clearInterval(interval);
  }, [isConnected, isProcessing, getStats]);

  return {
    // Connection state
    isConnected,
    isProcessing,
    sessionId,
    error,
    
    // Data
    currentPose,
    gaitMetrics,
    statistics,
    config,
    
    // Actions
    connect,
    disconnect,
    sendFrame,
    updateConfig,
    getStats
  };
}