/**
 * Navigation Type Definitions
 * 
 * Defines TypeScript interfaces and types for the navigation system,
 * following OOP principles with clear type hierarchies and contracts.
 * 
 * @module lib/navigation-types
 */

/**
 * Represents a navigation item in the application menu.
 * 
 * This interface defines the structure of navigation items used throughout
 * the application, supporting hierarchical navigation with nested children.
 */
export interface NavigationItem {
  /** Unique identifier for the navigation item */
  id: string;
  
  /** Display label for the navigation item */
  label: string;
  
  /** URL path for the navigation item */
  href: string;
  
  /** Optional icon name (from lucide-react) */
  icon?: string;
  
  /** Optional description for tooltips */
  description?: string;
  
  /** Whether this item is currently active */
  active?: boolean;
  
  /** Whether this item is disabled */
  disabled?: boolean;
  
  /** Badge text to display (e.g., "New", "Beta") */
  badge?: string;
  
  /** Badge variant for styling */
  badgeVariant?: 'default' | 'secondary' | 'destructive' | 'outline';
  
  /** Nested navigation items */
  children?: NavigationItem[];
  
  /** Whether children should be shown */
  expanded?: boolean;
  
  /** Access control - required roles */
  requiredRoles?: string[];
  
  /** Whether this item should be shown in mobile menu */
  showInMobile?: boolean;
  
  /** Whether this item should be shown in desktop menu */
  showInDesktop?: boolean;
  
  /** External link indicator */
  external?: boolean;
  
  /** Custom metadata */
  metadata?: Record<string, any>;
}

/**
 * Represents a user in the navigation system.
 * 
 * This interface defines user information displayed in navigation components
 * like user menus and profile dropdowns.
 */
export interface User {
  /** Unique user identifier */
  id: string;
  
  /** User's display name */
  name: string;
  
  /** User's email address */
  email: string;
  
  /** URL to user's avatar image */
  avatar?: string;
  
  /** User's role(s) in the system */
  roles: string[];
  
  /** User's initials for avatar fallback */
  initials?: string;
  
  /** Whether user is authenticated */
  authenticated: boolean;
  
  /** User preferences */
  preferences?: UserPreferences;
}

/**
 * User preferences for UI customization.
 */
export interface UserPreferences {
  /** Preferred theme */
  theme: Theme;
  
  /** Preferred language */
  language: string;
  
  /** Notification settings */
  notifications: boolean;
  
  /** Custom settings */
  custom?: Record<string, any>;
}

/**
 * Theme options for the application.
 */
export type Theme = 'light' | 'dark' | 'system';

/**
 * Navigation configuration structure.
 * 
 * Defines the complete navigation structure for the application,
 * including main navigation, user menu, and footer links.
 */
export interface NavigationConfig {
  /** Main navigation items */
  main: NavigationItem[];
  
  /** User menu items */
  user?: NavigationItem[];
  
  /** Footer navigation items */
  footer?: NavigationItem[];
  
  /** Mobile-specific navigation items */
  mobile?: NavigationItem[];
  
  /** Quick actions/shortcuts */
  quickActions?: NavigationItem[];
}

/**
 * Breadcrumb item for navigation trails.
 */
export interface BreadcrumbItem {
  /** Display label */
  label: string;
  
  /** URL path */
  href: string;
  
  /** Whether this is the current page */
  current?: boolean;
  
  /** Icon name */
  icon?: string;
}

/**
 * Navigation state for tracking current location.
 */
export interface NavigationState {
  /** Current pathname */
  pathname: string;
  
  /** Current breadcrumbs */
  breadcrumbs: BreadcrumbItem[];
  
  /** Active navigation item */
  activeItem: NavigationItem | null;
  
  /** Navigation history */
  history: string[];
}

/**
 * Navigation context for React context API.
 */
export interface NavigationContextValue {
  /** Navigation configuration */
  config: NavigationConfig;
  
  /** Current navigation state */
  state: NavigationState;
  
  /** Current user */
  user: User | null;
  
  /** Current theme */
  theme: Theme;
  
  /** Set theme function */
  setTheme: (theme: Theme) => void;
  
  /** Navigate function */
  navigate: (href: string) => void;
  
  /** Check if user has required roles */
  hasRole: (roles: string[]) => boolean;
}

/**
 * Navigation menu props.
 */
export interface NavigationMenuProps {
  /** Navigation items to display */
  items: NavigationItem[];
  
  /** Current pathname */
  pathname?: string;
  
  /** Orientation of the menu */
  orientation?: 'horizontal' | 'vertical';
  
  /** Whether to show icons */
  showIcons?: boolean;
  
  /** Whether to show badges */
  showBadges?: boolean;
  
  /** Custom className */
  className?: string;
  
  /** Callback when item is clicked */
  onItemClick?: (item: NavigationItem) => void;
}

/**
 * Breadcrumb props.
 */
export interface BreadcrumbProps {
  /** Breadcrumb items */
  items: BreadcrumbItem[];
  
  /** Separator character or element */
  separator?: React.ReactNode;
  
  /** Custom className */
  className?: string;
  
  /** Maximum items to show before collapsing */
  maxItems?: number;
}
