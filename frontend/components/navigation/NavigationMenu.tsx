/**
 * Navigation Menu with dropdown submenus
 * Features: hover effects, keyboard navigation, tooltips
 * 
 * Note: Radix UI NavigationMenu generates random IDs that differ between SSR and client.
 * This is expected behavior and doesn't affect functionality.
 */

'use client';

import React from 'react';
import Link from 'next/link';
import {
  NavigationMenu as NavMenu,
  NavigationMenuContent,
  NavigationMenuItem,
  NavigationMenuList,
  NavigationMenuTrigger,
} from '@/components/ui/navigation-menu';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { Badge } from '@/components/ui/badge';
import { NavigationItem } from '@/applib/navigation-types';

interface NavigationMenuProps {
  items: NavigationItem[];
  activeItem: string | null;
}

export function NavigationMenu({ items, activeItem }: NavigationMenuProps) {
  return (
    <TooltipProvider delayDuration={500}>
      <NavMenu suppressHydrationWarning>
        <NavigationMenuList>
          {items.map((item) => (
            <NavigationMenuItem key={item.id} suppressHydrationWarning>
              {item.children ? (
                <>
                  <NavigationMenuTrigger
                    className={`h-10 px-4 py-2 rounded-md transition-colors flex items-center whitespace-nowrap ${
                      activeItem === item.id
                        ? 'bg-accent text-accent-foreground'
                        : 'hover:bg-accent/50'
                    }`}
                  >
                    <span className="whitespace-nowrap">{item.label}</span>
                    {item.badge && (
                      <Badge variant="secondary" className="ml-2">
                        {item.badge}
                      </Badge>
                    )}
                  </NavigationMenuTrigger>
                  <NavigationMenuContent>
                    <div className="grid gap-3 p-6 w-[400px] lg:w-[500px] lg:grid-cols-2">
                      {item.children.map((subitem) => (
                        <Link
                          key={subitem.id}
                          href={subitem.href}
                          className="group block select-none space-y-1 rounded-md p-3 leading-none no-underline outline-none transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground"
                        >
                          <div className="flex items-center space-x-2">
                            <div className="text-sm font-medium leading-none">
                              {subitem.label}
                              {subitem.badge && (
                                <Badge variant="outline" className="ml-2 text-xs">
                                  {subitem.badge}
                                </Badge>
                              )}
                            </div>
                          </div>
                          <p className="line-clamp-2 text-sm leading-snug text-muted-foreground">
                            {subitem.description}
                          </p>
                        </Link>
                      ))}
                    </div>
                  </NavigationMenuContent>
                </>
              ) : (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      href={item.href}
                      className={`inline-flex flex-row h-10 w-max items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors focus:outline-none disabled:pointer-events-none disabled:opacity-50 whitespace-nowrap ${
                        activeItem === item.id
                          ? 'bg-accent text-accent-foreground'
                          : 'hover:bg-accent/50'
                      }`}
                    >
                      <span className="whitespace-nowrap">{item.label}</span>
                      {item.badge && (
                        <Badge variant="secondary" className="ml-2">
                          {item.badge}
                        </Badge>
                      )}
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">{item.description}</p>
                  </TooltipContent>
                </Tooltip>
              )}
            </NavigationMenuItem>
          ))}
        </NavigationMenuList>
      </NavMenu>
    </TooltipProvider>
  );
}
