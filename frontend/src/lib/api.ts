const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api/v1";

interface LoginCredentials {
  email: string;
  password: string;
}

interface RegisterData {
  email: string;
  username: string;
  password: string;
  full_name?: string;
}

export interface User {
  id: number;
  email: string;
  username: string;
  full_name: string | null;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
}

interface TokenResponse {
  access_token: string;
  token_type: string;
  user: User;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        detail: response.statusText,
      }));
      throw new Error(error.detail || "Request failed");
    }

    return response.json();
  }

  private getAuthHeaders(token: string): HeadersInit {
    return {
      Authorization: `Bearer ${token}`,
    };
  }

  async login(credentials: LoginCredentials): Promise<TokenResponse> {
    return this.request<TokenResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(credentials),
    });
  }

  async register(data: RegisterData): Promise<TokenResponse> {
    return this.request<TokenResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async getProfile(token: string): Promise<User> {
    return this.request<User>("/auth/me", {
      headers: this.getAuthHeaders(token),
    });
  }

  async updateProfile(
    token: string,
    data: { username?: string; full_name?: string }
  ): Promise<User> {
    return this.request<User>("/auth/me", {
      method: "PATCH",
      headers: this.getAuthHeaders(token),
      body: JSON.stringify(data),
    });
  }
}

export const api = new ApiClient(API_BASE);
