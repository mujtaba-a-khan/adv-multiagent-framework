"use client";

import { useState } from "react";
import { Eye, EyeOff, Save } from "lucide-react";
import { toast } from "sonner";
import { Header } from "@/components/layout/header";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";

export default function SettingsPage() {
  const [showKeys, setShowKeys] = useState(false);
  const [ollamaUrl, setOllamaUrl] = useState("http://localhost:11434");
  const [openaiKey, setOpenaiKey] = useState("");
  const [anthropicKey, setAnthropicKey] = useState("");
  const [googleKey, setGoogleKey] = useState("");
  const [darkMode, setDarkMode] = useState(true);

  const handleSave = () => {
    toast.success("Settings saved");
  };

  return (
    <>
      <Header title="Settings" />

      <div className="flex flex-1 flex-col gap-6 p-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Settings</h2>
          <p className="text-muted-foreground">
            Configure providers, API keys, and defaults.
          </p>
        </div>

        <div className="mx-auto w-full max-w-2xl space-y-6">
          {/* Ollama config */}
          <Card>
            <CardHeader>
              <CardTitle>Ollama Configuration</CardTitle>
              <CardDescription>
                Local LLM provider — the primary engine for all agent roles.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="ollama_url">Ollama Base URL</Label>
                <Input
                  id="ollama_url"
                  value={ollamaUrl}
                  onChange={(e) => setOllamaUrl(e.target.value)}
                  placeholder="http://localhost:11434"
                />
                <p className="text-xs text-muted-foreground">
                  The HTTP endpoint where Ollama is running.
                </p>
              </div>
            </CardContent>
          </Card>

          {/* API Keys */}
          <Card>
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle>API Keys</CardTitle>
                  <CardDescription>
                    Optional commercial provider credentials.
                  </CardDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowKeys(!showKeys)}
                >
                  {showKeys ? (
                    <EyeOff className="mr-2 h-4 w-4" />
                  ) : (
                    <Eye className="mr-2 h-4 w-4" />
                  )}
                  {showKeys ? "Hide" : "Show"}
                </Button>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <APIKeyField
                label="OpenAI API Key"
                value={openaiKey}
                onChange={setOpenaiKey}
                show={showKeys}
                placeholder="sk-..."
              />
              <Separator />
              <APIKeyField
                label="Anthropic API Key"
                value={anthropicKey}
                onChange={setAnthropicKey}
                show={showKeys}
                placeholder="sk-ant-..."
              />
              <Separator />
              <APIKeyField
                label="Google AI API Key"
                value={googleKey}
                onChange={setGoogleKey}
                show={showKeys}
                placeholder="AI..."
              />
            </CardContent>
          </Card>

          {/* Defaults */}
          <Card>
            <CardHeader>
              <CardTitle>Defaults</CardTitle>
              <CardDescription>
                Default values for new experiments.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <Label>Dark Mode</Label>
                  <p className="text-xs text-muted-foreground">
                    Use dark theme by default.
                  </p>
                </div>
                <Switch checked={darkMode} onCheckedChange={setDarkMode} />
              </div>
            </CardContent>
          </Card>

          <div className="flex justify-end">
            <Button onClick={handleSave}>
              <Save className="mr-2 h-4 w-4" />
              Save Settings
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}

function APIKeyField({
  label,
  value,
  onChange,
  show,
  placeholder,
}: Readonly<{
  label: string;
  value: string;
  onChange: (val: string) => void;
  show: boolean;
  placeholder: string;
}>) {
  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <Input
        type={show ? "text" : "password"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
      />
    </div>
  );
}
