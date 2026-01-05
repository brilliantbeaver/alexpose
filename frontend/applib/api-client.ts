/**
 * API Client Module
 * 
 * Provides a centralized, type-safe API client for communicating with the backend.
 * Implements error handling, request/response interceptors, and retry logic.
 * 
 * @module lib/api-client
 */

/**
 * API Error class for structured error handling.
 */
export class APIError extends Error {
  /** HTTP status code */
  public readonly status: number;
  
  /** Error code from API */
  public readonly code?: string;
  
  /** Additional error details */
  public readonly details?: any;
  
  /** Original response */
  public readonly response?: Response;
  
  constructor(
    message: string,
    status: number,
    code?: string,
    details?: any,
    response?: Response
  ) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.code = code;
    this.details = details;
    this.response = response;
    
    // Maintains proper stack trace for where error was thrown (V8 only)
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, APIError);
    }
  }
  
  /**
   * Check if error is a specific HTTP status.
   */
  public isStatus(status: number): boolean {
    return this.status === status;
  }
  
  /**
   * Check if error is a client error (4xx).
   */
  public isClientError(): boolean {
    return this.status >= 400 && this.status < 500;
  }
  
  /**
   * Check if error is a server error (5xx).
   */
  public isServerError(): boolean {
    return this.status >= 500 && this.status < 600;
  }
}

/**
 * API client configuration.
 */
export interface APIClientConfig {
  /** Base URL for API requests */
  baseURL?: string;
  
  /** Default headers */
  headers?: Record<string, string>;
  
  /** Request timeout in milliseconds */
  timeout?: number;
  
  /** Number of retry attempts */
  retries?: number;
  
  /** Retry delay in milliseconds */
  retryDelay?: number;
  
  /** Whether to include credentials */
  credentials?: RequestCredentials;
}

/**
 * Request options.
 */
export interface RequestOptions extends RequestInit {
  /** Query parameters */
  params?: Record<string, string | number | boolean>;
  
  /** Request timeout */
  timeout?: number;
  
  /** Number of retries */
  retries?: number;
  
  /** Whether to parse response as JSON */
  parseJSON?: boolean;
}

/**
 * API Client class for making HTTP requests.
 * 
 * Provides a clean, type-safe interface for API communication with
 * built-in error handling, retries, and request/response processing.
 */
export class APIClient {
  private config: Required<APIClientConfig>;
  
  constructor(config: APIClientConfig = {}) {
    this.config = {
      baseURL: config.baseURL || '/api',
      headers: config.headers || {
        'Content-Type': 'application/json',
      },
      timeout: config.timeout || 30000,
      retries: config.retries || 3,
      retryDelay: config.retryDelay || 1000,
      credentials: config.credentials || 'same-origin',
    };
  }
  
  /**
   * Make a GET request.
   */
  public async get<T = any>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'GET',
    });
  }
  
  /**
   * Make a POST request.
   */
  public async post<T = any>(
    endpoint: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'POST',
      body: data ? JSON.stringify(data) : undefined,
    });
  }
  
  /**
   * Make a PUT request.
   */
  public async put<T = any>(
    endpoint: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PUT',
      body: data ? JSON.stringify(data) : undefined,
    });
  }
  
  /**
   * Make a PATCH request.
   */
  public async patch<T = any>(
    endpoint: string,
    data?: any,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'PATCH',
      body: data ? JSON.stringify(data) : undefined,
    });
  }
  
  /**
   * Make a DELETE request.
   */
  public async delete<T = any>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    return this.request<T>(endpoint, {
      ...options,
      method: 'DELETE',
    });
  }
  
  /**
   * Make a generic HTTP request with retry logic.
   */
  private async request<T>(
    endpoint: string,
    options: RequestOptions = {}
  ): Promise<T> {
    const {
      params,
      timeout = this.config.timeout,
      retries = this.config.retries,
      parseJSON = true,
      ...fetchOptions
    } = options;
    
    // Build URL with query parameters
    const url = this.buildURL(endpoint, params);
    
    // Merge headers
    const headers = {
      ...this.config.headers,
      ...fetchOptions.headers,
    };
    
    // Attempt request with retries
    let lastError: Error | null = null;
    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const response = await this.fetchWithTimeout(url, {
          ...fetchOptions,
          headers,
          credentials: this.config.credentials,
        }, timeout);
        
        // Handle response
        return await this.handleResponse<T>(response, parseJSON);
      } catch (error) {
        lastError = error as Error;
        
        // Don't retry on client errors (4xx)
        if (error instanceof APIError && error.isClientError()) {
          throw error;
        }
        
        // Wait before retry (except on last attempt)
        if (attempt < retries) {
          await this.delay(this.config.retryDelay * (attempt + 1));
        }
      }
    }
    
    // All retries failed
    throw lastError || new Error('Request failed after retries');
  }
  
  /**
   * Fetch with timeout support.
   */
  private async fetchWithTimeout(
    url: string,
    options: RequestInit,
    timeout: number
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      return response;
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        throw new APIError('Request timeout', 408, 'TIMEOUT');
      }
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }
  
  /**
   * Handle API response.
   */
  private async handleResponse<T>(
    response: Response,
    parseJSON: boolean
  ): Promise<T> {
    // Success responses
    if (response.ok) {
      if (response.status === 204 || !parseJSON) {
        return undefined as T;
      }
      
      try {
        return await response.json();
      } catch (error) {
        throw new APIError(
          'Failed to parse response JSON',
          response.status,
          'PARSE_ERROR',
          error,
          response
        );
      }
    }
    
    // Error responses
    let errorData: any;
    try {
      errorData = await response.json();
    } catch {
      errorData = { message: response.statusText };
    }
    
    throw new APIError(
      errorData.message || errorData.detail || 'Request failed',
      response.status,
      errorData.code,
      errorData,
      response
    );
  }
  
  /**
   * Build URL with query parameters.
   */
  private buildURL(
    endpoint: string,
    params?: Record<string, string | number | boolean>
  ): string {
    const baseURL = this.config.baseURL;
    const url = endpoint.startsWith('http')
      ? endpoint
      : `${baseURL}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    
    if (!params || Object.keys(params).length === 0) {
      return url;
    }
    
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      searchParams.append(key, String(value));
    });
    
    return `${url}?${searchParams.toString()}`;
  }
  
  /**
   * Delay helper for retries.
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Default API client instance.
 */
export const apiClient = new APIClient({
  baseURL: process.env.NEXT_PUBLIC_API_URL || '/api',
});

/**
 * Create a custom API client with specific configuration.
 */
export function createAPIClient(config: APIClientConfig): APIClient {
  return new APIClient(config);
}
