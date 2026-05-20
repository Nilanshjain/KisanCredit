'use client'

import React from 'react'
import { cn } from '@/lib/utils'

export interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: 'text' | 'title' | 'avatar' | 'rectangular' | 'circular'
  width?: string | number
  height?: string | number
}

const Skeleton = React.forwardRef<HTMLDivElement, SkeletonProps>(
  ({ className, variant = 'rectangular', width, height, style, ...props }, ref) => {
    const baseClasses = 'animate-pulse bg-stone-200'

    const variants = {
      text: 'h-4 rounded-xl',
      title: 'h-8 rounded-xl',
      avatar: 'h-12 w-12 rounded-full',
      rectangular: 'rounded-xl',
      circular: 'rounded-full',
    }

    const customStyle: React.CSSProperties = {
      ...style,
      ...(width && { width: typeof width === 'number' ? `${width}px` : width }),
      ...(height && { height: typeof height === 'number' ? `${height}px` : height }),
    }

    return (
      <div
        ref={ref}
        className={cn(baseClasses, variants[variant], className)}
        style={customStyle}
        {...props}
      />
    )
  }
)

Skeleton.displayName = 'Skeleton'

export default Skeleton
