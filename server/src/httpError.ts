// An Error carrying the HTTP-ish status code the WebSocket layer reports back to the requester
// (wsServer.ts's dispatch site reads `statusCode` off whatever a handler throws, defaulting to 500).
export function httpError(statusCode: number, message: string): Error & { statusCode: number } {
    return Object.assign(new Error(message), { statusCode });
}
