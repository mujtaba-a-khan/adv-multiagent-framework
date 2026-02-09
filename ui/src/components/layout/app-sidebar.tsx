"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  BarChart3,
  Database,
  FlaskConical,
  LayoutDashboard,
  Server,
  Settings,
  Shield,
  Swords,
  Wrench,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar";
import { ROUTES } from "@/lib/constants";

const NAV_ITEMS = [
  {
    label: "Overview",
    items: [
      {
        title: "Dashboard",
        href: ROUTES.dashboard,
        icon: LayoutDashboard,
      },
    ],
  },
  {
    label: "Red Teaming",
    items: [
      {
        title: "Experiments",
        href: ROUTES.experiments.list,
        icon: FlaskConical,
      },
      {
        title: "Reports",
        href: ROUTES.reports.list,
        icon: BarChart3,
      },
      {
        title: "Strategies",
        href: ROUTES.strategies,
        icon: Swords,
      },
    ],
  },
  {
    label: "Configuration",
    items: [
      {
        title: "Model Workshop",
        href: ROUTES.workshop.list,
        icon: Wrench,
      },
      {
        title: "Abliteration Dataset",
        href: ROUTES.workshop.dataset,
        icon: Database,
      },
      {
        title: "Targets & Models",
        href: ROUTES.targets,
        icon: Server,
      },
      {
        title: "Settings",
        href: ROUTES.settings,
        icon: Settings,
      },
    ],
  },
];

export function AppSidebar() {
  const pathname = usePathname();

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname.startsWith(href);
  };

  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader className="border-b border-sidebar-border">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link href="/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <Shield className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">
                    Adversarial AI
                  </span>
                  <span className="truncate text-xs text-muted-foreground">
                    Safety Framework
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        {NAV_ITEMS.map((group) => (
          <SidebarGroup key={group.label}>
            <SidebarGroupLabel>{group.label}</SidebarGroupLabel>
            <SidebarGroupContent>
              <SidebarMenu>
                {group.items.map((item) => (
                  <SidebarMenuItem key={item.title}>
                    <SidebarMenuButton
                      asChild
                      isActive={isActive(item.href)}
                      tooltip={item.title}
                    >
                      <Link href={item.href}>
                        <item.icon />
                        <span>{item.title}</span>
                      </Link>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>

      <SidebarRail />
    </Sidebar>
  );
}
