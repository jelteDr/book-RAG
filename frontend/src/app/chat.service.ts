import { Injectable } from '@angular/core';

import { ChatEvent, ChatRequest, ModelsResponse } from './models';

/** Kapselt die HTTP-/SSE-Kommunikation mit dem Backend (via /api-Proxy). */
@Injectable({ providedIn: 'root' })
export class ChatService {
  async getModels(): Promise<ModelsResponse> {
    const resp = await fetch('/api/models');
    if (!resp.ok) throw new Error(`/api/models: ${resp.status}`);
    return resp.json();
  }

  /** Streamt die Antwort als Folge von ChatEvents (Token, dann done bzw. error). */
  async *stream(request: ChatRequest): AsyncGenerator<ChatEvent> {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    if (!resp.body) throw new Error('Kein Response-Body');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split('\n\n');
      buffer = parts.pop() ?? '';
      for (const part of parts) {
        const event = this.parseEvent(part);
        if (event) yield event;
      }
    }
  }

  private parseEvent(part: string): ChatEvent | null {
    let event = '';
    let data = '';
    for (const line of part.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data = line.slice(5).trim();
    }
    if (!data) return null;
    return { event: event as ChatEvent['event'], data: JSON.parse(data) };
  }
}
