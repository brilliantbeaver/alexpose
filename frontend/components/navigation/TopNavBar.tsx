/**
 * Top Navigation Bar with Liquid Glass design
 * Features: responsive layout, theme toggle, user profile, glassmorphism effects
 * 
 * Note: Radix UI components generate random IDs that differ between SSR and client.
 * suppressHydrationWarning is used to prevent console warnings without affecting functionality.
 */

'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { useNavigation } from '@/hooks/useNavigation';
import { NavigationMenu } from './NavigationMenu';
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { User, Theme } from '@/lib/navigation-types';
import { Sun, Moon, Menu, Settings, User as UserIcon, CreditCard, LogOut } from 'lucide-react';

interface TopNavBarProps {
  user?: User;
  theme?: Theme;
  onThemeToggle?: () => void;
}

export function TopNavBar({ user, theme = 'light', onThemeToggle }: TopNavBarProps) {
  const { navigationItems, activeItem } = useNavigation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav 
      className="sticky top-0 z-50 w-full border-b border-border/20 bg-background/80 backdrop-blur-xl supports-[backdrop-filter]:bg-background/70 shadow-sm transition-all duration-300" 
      suppressHydrationWarning
    >
      <div className="container flex h-16 items-center px-4 gap-4">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-3 group">
          <div className="flex items-center justify-center w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 text-white font-bold text-xl shadow-lg group-hover:shadow-xl group-hover:scale-105 transition-all duration-300">
            AP
          </div>
          <span className="hidden font-bold sm:inline-block text-xl bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent group-hover:from-blue-500 group-hover:to-purple-500 transition-all duration-300">
            AlexPose
          </span>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex flex-1">
          <NavigationMenu items={navigationItems} activeItem={activeItem} />
        </div>

        {/* Right side actions */}
        <div className="flex items-center gap-2 ml-auto">
          {/* Theme Toggle */}
          <Button
            variant="ghost"
            size="icon"
            onClick={onThemeToggle}
            className="rounded-full hover:bg-accent/80 hover:scale-105 transition-all duration-200"
            aria-label="Toggle theme"
          >
            {theme === 'dark' ? (
              <Moon className="w-5 h-5" />
            ) : (
              <Sun className="w-5 h-5" />
            )}
          </Button>

          {/* User Profile */}
          {user ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild suppressHydrationWarning>
                <Button
                  variant="ghost"
                  className="relative h-10 w-10 rounded-full hover:bg-accent/80 hover:scale-105 transition-all duration-200"
                >
                  <div className="flex items-center justify-center w-full h-full rounded-full bg-gradient-to-br from-blue-400 to-purple-500 text-white font-semibold shadow-md">
                    {user.name.charAt(0).toUpperCase()}
                  </div>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56 backdrop-blur-xl bg-background/95">
                <DropdownMenuLabel>
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium leading-none">{user.name}</p>
                    <p className="text-xs leading-none text-muted-foreground">
                      {user.email}
                    </p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem asChild>
                  <Link href="/settings" className="flex items-center gap-2 cursor-pointer">
                    <Settings className="w-4 h-4" />
                    <span>Settings</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/account" className="flex items-center gap-2 cursor-pointer">
                    <UserIcon className="w-4 h-4" />
                    <span>Account</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuItem asChild>
                  <Link href="/billing" className="flex items-center gap-2 cursor-pointer">
                    <CreditCard className="w-4 h-4" />
                    <span>Billing</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem className="text-red-600 flex items-center gap-2 cursor-pointer">
                  <LogOut className="w-4 h-4" />
                  <span>Logout</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Button asChild variant="default" className="rounded-full hover:scale-105 transition-all duration-200 shadow-md hover:shadow-lg">
              <Link href="/login">Sign In</Link>
            </Button>
          )}

          {/* Mobile Menu Toggle */}
          <Sheet open={mobileMenuOpen} onOpenChange={setMobileMenuOpen}>
            <SheetTrigger asChild className="md:hidden" suppressHydrationWarning>
              <Button variant="ghost" size="icon" className="rounded-full hover:bg-accent/80 hover:scale-105 transition-all duration-200">
                <Menu className="w-5 h-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-80 backdrop-blur-xl bg-background/95">
              <div className="flex flex-col space-y-4 mt-8">
                <div className="text-lg font-semibold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  Navigation
                </div>
                {navigationItems.map((item) => (
                  <div key={item.id} className="space-y-2">
                    <Link
                      href={item.href}
                      className={`flex items-center gap-3 px-4 py-2 rounded-lg transition-all duration-200 ${
                        activeItem === item.id
                          ? 'bg-accent text-accent-foreground shadow-sm'
                          : 'hover:bg-accent/50 hover:translate-x-1'
                      }`}
                      onClick={() => setMobileMenuOpen(false)}
                    >
                      {item.icon && <item.icon className="w-5 h-5" />}
                      <span className="font-medium">{item.label}</span>
                    </Link>
                    {item.submenu && (
                      <div className="ml-8 space-y-1">
                        {item.submenu.map((subitem) => (
                          <Link
                            key={subitem.id}
                            href={subitem.href}
                            className="flex items-center gap-2 px-4 py-1.5 text-sm rounded-md hover:bg-accent/50 hover:translate-x-1 transition-all duration-200"
                            onClick={() => setMobileMenuOpen(false)}
                          >
                            {subitem.icon && <subitem.icon className="w-4 h-4" />}
                            <span>{subitem.label}</span>
                            {subitem.comingSoon && (
                              <span className="text-xs text-muted-foreground">(Soon)</span>
                            )}
                          </Link>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </nav>
  );
}
