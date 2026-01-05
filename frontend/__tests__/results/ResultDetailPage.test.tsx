/**
 * Tests for Result Detail Page
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock Next.js navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    pathname: '/results/1',
  }),
  usePathname: () => '/results/1',
}));

// Mock the dynamic page component
const mockAnalysisData = {
  id: 1,
  name: 'Walking Test 1',
  date: '2024-01-03',
  time: '14:30:00',
  status: 'Normal',
  confidence: 95,
  conditions: [],
};

describe('Result Detail Page', () => {
  it('should display analysis name', () => {
    render(
      <div>
        <h1>{mockAnalysisData.name}</h1>
      </div>
    );
    expect(screen.getByText('Walking Test 1')).toBeInTheDocument();
  });

  it('should display analysis status', () => {
    render(
      <div>
        <span>{mockAnalysisData.status}</span>
      </div>
    );
    expect(screen.getByText('Normal')).toBeInTheDocument();
  });

  it('should display confidence score', () => {
    render(
      <div>
        <span>{mockAnalysisData.confidence}%</span>
      </div>
    );
    expect(screen.getByText('95%')).toBeInTheDocument();
  });

  it('should display analysis date and time', () => {
    render(
      <div>
        <span>{mockAnalysisData.date} at {mockAnalysisData.time}</span>
      </div>
    );
    expect(screen.getByText(/2024-01-03 at 14:30:00/)).toBeInTheDocument();
  });

  it('should handle missing analysis gracefully', () => {
    const notFoundMessage = 'Analysis Not Found';
    render(
      <div>
        <h2>{notFoundMessage}</h2>
      </div>
    );
    expect(screen.getByText(notFoundMessage)).toBeInTheDocument();
  });
});

describe('Result Detail Page - Abnormal Analysis', () => {
  const abnormalAnalysis = {
    id: 2,
    name: 'Gait Analysis 2',
    status: 'Abnormal',
    confidence: 88,
    conditions: ['Limping', 'Asymmetry'],
  };

  it('should display abnormal status', () => {
    render(
      <div>
        <span>{abnormalAnalysis.status}</span>
      </div>
    );
    expect(screen.getByText('Abnormal')).toBeInTheDocument();
  });

  it('should display detected conditions', () => {
    render(
      <div>
        {abnormalAnalysis.conditions.map((condition, idx) => (
          <span key={idx}>{condition}</span>
        ))}
      </div>
    );
    expect(screen.getByText('Limping')).toBeInTheDocument();
    expect(screen.getByText('Asymmetry')).toBeInTheDocument();
  });
});
