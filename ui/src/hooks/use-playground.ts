import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listPlaygroundConversations,
  getPlaygroundConversation,
  createPlaygroundConversation,
  updatePlaygroundConversation,
  deletePlaygroundConversation,
  listPlaygroundMessages,
  sendPlaygroundMessage,
} from "@/lib/api-client";
import type {
  CreatePlaygroundConversationRequest,
  UpdatePlaygroundConversationRequest,
} from "@/lib/types";

export function usePlaygroundConversations() {
  return useQuery({
    queryKey: ["playground-conversations"],
    queryFn: () => listPlaygroundConversations(0, 100),
  });
}

export function usePlaygroundConversation(id: string) {
  return useQuery({
    queryKey: ["playground-conversation", id],
    queryFn: () => getPlaygroundConversation(id),
    enabled: !!id,
  });
}

export function useCreatePlaygroundConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (data: CreatePlaygroundConversationRequest) =>
      createPlaygroundConversation(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["playground-conversations"] });
    },
  });
}

export function useUpdatePlaygroundConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      data,
    }: {
      id: string;
      data: UpdatePlaygroundConversationRequest;
    }) => updatePlaygroundConversation(id, data),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ["playground-conversations"] });
      qc.invalidateQueries({
        queryKey: ["playground-conversation", variables.id],
      });
    },
  });
}

export function useDeletePlaygroundConversation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => deletePlaygroundConversation(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["playground-conversations"] });
    },
  });
}

export function usePlaygroundMessages(conversationId: string) {
  return useQuery({
    queryKey: ["playground-messages", conversationId],
    queryFn: () => listPlaygroundMessages(conversationId, 0, 500),
    enabled: !!conversationId,
  });
}

export function useSendPlaygroundMessage() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      conversationId,
      prompt,
    }: {
      conversationId: string;
      prompt: string;
    }) => sendPlaygroundMessage(conversationId, prompt),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({
        queryKey: ["playground-messages", variables.conversationId],
      });
      qc.invalidateQueries({
        queryKey: ["playground-conversation", variables.conversationId],
      });
      qc.invalidateQueries({ queryKey: ["playground-conversations"] });
    },
  });
}
