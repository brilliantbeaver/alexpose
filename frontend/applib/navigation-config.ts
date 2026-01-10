/**
 * Navigation Configuration
 * 
 * Centralized navigation configuration for the AlexPose application.
 * Defines all navigation items, menus, and routing structure.
 * 
 * @module lib/navigation-config
 */

import { NavigationConfig, NavigationItem } from './navigation-types';

/**
 * Main navigation items for the application.
 * 
 * These items appear in the top navigation bar and define the primary
 * navigation structure of the application.
 */
const mainNavigation: NavigationItem[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    href: '/dashboard',
    icon: 'LayoutDashboard',
    description: 'Overview and quick access',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'analyses',
    label: 'Analyses',
    href: '/analyses',
    icon: 'FileText',
    description: 'View all your analyses',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'analyze',
    label: 'Analyze',
    href: '/analyze',
    icon: 'Activity',
    description: 'Analyze gait videos',
    showInMobile: true,
    showInDesktop: true,
    children: [
      {
        id: 'analyze-gavd',
        label: 'GAVD Dataset Analysis',
        href: '/gavd',
        description: 'Analyze videos from GAVD clinical dataset',
        showInMobile: true,
        showInDesktop: true,
      },
      {
        id: 'analyze-upload',
        label: 'Upload Video',
        href: '/analyze/upload',
        description: 'Upload and analyze your own video files',
        showInMobile: true,
        showInDesktop: true,
      },
      {
        id: 'analyze-youtube',
        label: 'YouTube Analysis',
        href: '/analyze/youtube',
        description: 'Analyze gait videos from YouTube URLs',
        showInMobile: true,
        showInDesktop: true,
      },
    ],
  },
  {
    id: 'models',
    label: 'Models',
    href: '/models',
    icon: 'Brain',
    description: 'AI models and configurations',
    showInMobile: true,
    showInDesktop: true,
    children: [
      {
        id: 'models-explore',
        label: 'Explore Models',
        href: '/models/browse',
        description: 'Browse pose estimation and LLM models',
        showInMobile: true,
        showInDesktop: true,
      },
    ],
  },
  {
    id: 'help',
    label: 'Help',
    href: '/help',
    icon: 'HelpCircle',
    description: 'Documentation and support',
    showInMobile: true,
    showInDesktop: true,
  },
];

/**
 * User menu items.
 * 
 * These items appear in the user dropdown menu in the top navigation bar.
 */
const userNavigation: NavigationItem[] = [
  {
    id: 'profile',
    label: 'Profile',
    href: '/profile',
    icon: 'User',
    description: 'View and edit your profile',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'settings',
    label: 'Settings',
    href: '/settings',
    icon: 'Settings',
    description: 'Application settings',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'billing',
    label: 'Billing',
    href: '/billing',
    icon: 'CreditCard',
    description: 'Manage billing and subscription',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'logout',
    label: 'Log out',
    href: '/logout',
    icon: 'LogOut',
    description: 'Sign out of your account',
    showInMobile: true,
    showInDesktop: true,
  },
];

/**
 * Footer navigation items.
 * 
 * These items appear in the application footer.
 */
const footerNavigation: NavigationItem[] = [
  {
    id: 'about',
    label: 'About',
    href: '/about',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'privacy',
    label: 'Privacy',
    href: '/privacy',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'terms',
    label: 'Terms',
    href: '/terms',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'contact',
    label: 'Contact',
    href: '/contact',
    showInMobile: true,
    showInDesktop: true,
  },
];

/**
 * Quick action items.
 * 
 * These items provide quick access to common actions.
 */
const quickActions: NavigationItem[] = [
  {
    id: 'new-analysis',
    label: 'New Analysis',
    href: '/analyze/new',
    icon: 'Plus',
    description: 'Start a new gait analysis',
    showInMobile: true,
    showInDesktop: true,
  },
  {
    id: 'upload-video',
    label: 'Upload Video',
    href: '/analyze/upload',
    icon: 'Upload',
    description: 'Upload a video for analysis',
    showInMobile: true,
    showInDesktop: true,
  },
];

/**
 * Complete navigation configuration for the application.
 * 
 * This configuration object is used throughout the application to render
 * navigation menus, breadcrumbs, and other navigation-related components.
 */
export const navigationConfig: NavigationConfig = {
  main: mainNavigation,
  user: userNavigation,
  footer: footerNavigation,
  quickActions: quickActions,
};

/**
 * Helper function to find a navigation item by ID.
 * 
 * @param id - Navigation item ID to find
 * @param items - Array of navigation items to search (defaults to main navigation)
 * @returns Found navigation item or undefined
 */
export function findNavigationItem(
  id: string,
  items: NavigationItem[] = mainNavigation
): NavigationItem | undefined {
  for (const item of items) {
    if (item.id === id) {
      return item;
    }
    if (item.children) {
      const found = findNavigationItem(id, item.children);
      if (found) {
        return found;
      }
    }
  }
  return undefined;
}

/**
 * Helper function to find a navigation item by href.
 * 
 * @param href - Navigation item href to find
 * @param items - Array of navigation items to search (defaults to main navigation)
 * @returns Found navigation item or undefined
 */
export function findNavigationItemByHref(
  href: string,
  items: NavigationItem[] = mainNavigation
): NavigationItem | undefined {
  for (const item of items) {
    if (item.href === href) {
      return item;
    }
    if (item.children) {
      const found = findNavigationItemByHref(href, item.children);
      if (found) {
        return found;
      }
    }
  }
  return undefined;
}

/**
 * Helper function to get breadcrumbs for a given path.
 * 
 * @param pathname - Current pathname
 * @returns Array of breadcrumb items
 */
export function getBreadcrumbs(pathname: string): Array<{ label: string; href: string }> {
  const segments = pathname.split('/').filter(Boolean);
  const breadcrumbs: Array<{ label: string; href: string }> = [
    { label: 'Home', href: '/' },
  ];
  
  let currentPath = '';
  for (const segment of segments) {
    currentPath += `/${segment}`;
    const item = findNavigationItemByHref(currentPath);
    
    if (item) {
      breadcrumbs.push({
        label: item.label,
        href: item.href,
      });
    } else {
      // Fallback: capitalize segment
      breadcrumbs.push({
        label: segment.charAt(0).toUpperCase() + segment.slice(1),
        href: currentPath,
      });
    }
  }
  
  return breadcrumbs;
}
