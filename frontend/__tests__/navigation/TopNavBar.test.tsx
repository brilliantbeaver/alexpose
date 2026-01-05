/**
 * Tests for TopNavBar component
 */

import { render, screen } from '@testing-library/react';
import { TopNavBar } from '@/components/navigation/TopNavBar';

// Mock next/navigation
jest.mock('next/navigation', () => ({
  usePathname: () => '/',
}));

describe('TopNavBar', () => {
  it('renders the logo', () => {
    render(<TopNavBar />);
    expect(screen.getByText('AlexPose')).toBeInTheDocument();
  });

  it('renders theme toggle button', () => {
    render(<TopNavBar theme="light" />);
    const themeButton = screen.getByLabelText('Toggle theme');
    expect(themeButton).toBeInTheDocument();
  });

  it('renders sign in button when no user', () => {
    render(<TopNavBar />);
    expect(screen.getByText('Sign In')).toBeInTheDocument();
  });

  it('renders user profile when user is provided', () => {
    const user = {
      id: '1',
      name: 'John Doe',
      email: 'john@example.com',
    };
    render(<TopNavBar user={user} />);
    expect(screen.getByText('J')).toBeInTheDocument(); // First letter of name
  });

  it('displays correct theme icon', () => {
    const { rerender } = render(<TopNavBar theme="light" />);
    expect(screen.getByText('☀️')).toBeInTheDocument();
    
    rerender(<TopNavBar theme="dark" />);
    expect(screen.getByText('🌙')).toBeInTheDocument();
  });
});
